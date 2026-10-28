use std::cmp::Ordering;
use std::collections::VecDeque;

use AsofStrategy::*;
use parking_lot::Mutex;
use polars_async::executor::{
    self, AbortOnDropHandle, JoinHandle, TaskMetricAggregator, TaskPriority, TaskScope,
};
use polars_async::primitives::distributor_channel as dc;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::prelude::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_defs::join::{AsOfOptions, AsofStrategy, JoinArgs, JoinType};
use polars_ooc::RandomSpillContext;
use polars_ops::frame::is_sorted::DataFrameIsSorted;
use polars_ops::frame::{_check_asof_columns, _finish_join, _join_asof_dispatch};
use polars_ops::series::{rle_lengths, rle_lengths_helper_ca};
use polars_utils::itertools::Itertools;
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::sort::reorder_cmp;
use tokio::sync::watch;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::ComputeNode;
use crate::nodes::joins::utils::{SpillFrameSearchBuffer, stop_and_take_pipe_contents};
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

const ROW_ENCODED_COL_NAME: PlSmallStr = PlSmallStr::from_static("__PL_ASOF_JOIN_BY");

#[derive(Debug)]
pub struct AsOfJoinSideParams {
    pub on: PlSmallStr,
    pub tmp_key_col: Option<PlSmallStr>,
}

impl AsOfJoinSideParams {
    fn key_col(&self) -> &PlSmallStr {
        self.tmp_key_col.as_ref().unwrap_or(&self.on)
    }
}

#[derive(Debug)]
struct AsOfJoinParams {
    left: AsOfJoinSideParams,
    right: AsOfJoinSideParams,
    by_descending: Vec<bool>,
    by_nulls_last: Vec<bool>,
    args: JoinArgs,
}

impl AsOfJoinParams {
    fn as_of_options(&self) -> &AsOfOptions {
        let JoinType::AsOf(ref options) = self.args.how else {
            unreachable!();
        };
        options
    }

    fn left_by(&self) -> &[PlSmallStr] {
        self.as_of_options()
            .left_by
            .as_ref()
            .map_or(&[], |x| &x[..])
    }

    fn right_by(&self) -> &[PlSmallStr] {
        self.as_of_options()
            .right_by
            .as_ref()
            .map_or(&[], |x| &x[..])
    }
}

#[derive(Debug, Default, PartialEq)]
enum AsOfJoinState {
    #[default]
    Running,
    FlushInputBuffer,
    Done,
}

#[derive(Debug)]
pub struct AsOfJoinNode {
    params: Arc<AsOfJoinParams>,
    state: AsOfJoinState,
    /// We may need to stash a morsel on the left side whenever we do not
    /// have enough data on the right side, but the right side is empty.
    /// In these cases, we stash that morsel here.
    left_buffer: VecDeque<DataFrame>,
    /// The left morsel being dispatched in pieces when a phase stopped, and the offset of its
    /// next piece.
    in_progress: Option<(Arc<MorselOutput>, usize)>,
    /// Buffer of the live range of right AsOf join rows.
    right_buffer: SpillFrameSearchBuffer,
    output_seq: MorselSeq,
    // Slots to store the last non-null row of the previous morsel.
    // Used to check that each side is sorted across morsel boundaries.
    last_non_null_row_left: Option<DataFrame>,
    last_non_null_row_right: Option<DataFrame>,
    /// Right rows at the end of the buffer, buffered when a phase stopped, that are checked
    /// for sortedness once a phase uses them.
    unchecked_right_rows: usize,
    /// The number of right rows checked for sortedness, counted across phases as a morsel's
    /// pieces may be cut in several.
    checked_right_rows: usize,
}

impl AsOfJoinNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        left_on: PlSmallStr,
        right_on: PlSmallStr,
        tmp_left_key_col: Option<PlSmallStr>,
        tmp_right_key_col: Option<PlSmallStr>,
        by_descending: Option<Vec<bool>>,
        by_nulls_last: Option<Vec<bool>>,
        args: JoinArgs,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        let JoinType::AsOf(ref options) = args.how else {
            unreachable!();
        };
        assert!({
            let by_len = || options.left_by.as_ref().unwrap().len();
            let by_all_none = options.left_by.is_none()
                && options.right_by.is_none()
                && by_descending.is_none()
                && by_nulls_last.is_none();
            by_all_none
                || (options.right_by.as_ref().unwrap().len() == by_len()
                    && by_descending.as_ref().unwrap().len() == by_len()
                    && by_nulls_last.as_ref().unwrap().len() == by_len())
        });

        let left_key_col = tmp_left_key_col.as_ref().unwrap_or(&left_on);
        let right_key_col = tmp_right_key_col.as_ref().unwrap_or(&right_on);
        let left_key_dtype = left_input_schema.get(left_key_col).unwrap();
        let right_key_dtype = right_input_schema.get(right_key_col).unwrap();
        assert_eq!(left_key_dtype, right_key_dtype);
        let left = AsOfJoinSideParams {
            on: left_on,
            tmp_key_col: tmp_left_key_col,
        };
        let right = AsOfJoinSideParams {
            on: right_on,
            tmp_key_col: tmp_right_key_col,
        };

        let params = AsOfJoinParams {
            left,
            right,
            by_descending: by_descending.unwrap_or_default(),
            by_nulls_last: by_nulls_last.unwrap_or_default(),
            args,
        };
        AsOfJoinNode {
            params: Arc::new(params),
            state: AsOfJoinState::default(),
            left_buffer: Default::default(),
            in_progress: None,
            right_buffer: SpillFrameSearchBuffer::empty_with_schema(
                right_input_schema,
                RandomSpillContext::new("asof-join-search-buffer".into(), task_metrics),
            ),
            output_seq: Default::default(),
            last_non_null_row_left: None,
            last_non_null_row_right: None,
            unchecked_right_rows: 0,
            checked_right_rows: 0,
        }
    }
}

impl ComputeNode for AsOfJoinNode {
    fn name(&self) -> &str {
        "asof-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        if send[0] == PortState::Done {
            self.state = AsOfJoinState::Done;
        }

        if self.state == AsOfJoinState::Running && recv[0] == PortState::Done {
            self.state = AsOfJoinState::FlushInputBuffer;
        }

        if self.state == AsOfJoinState::FlushInputBuffer
            && self.left_buffer.is_empty()
            && self.in_progress.is_none()
        {
            self.state = AsOfJoinState::Done;
        }

        let recv0_blocked = recv[0] == PortState::Blocked;
        let recv1_blocked = recv[1] == PortState::Blocked;
        let send_blocked = send[0] == PortState::Blocked;
        match self.state {
            AsOfJoinState::Running => {
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
                if recv0_blocked {
                    recv[1] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                }
                if recv1_blocked {
                    recv[0] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                }
                if send_blocked {
                    recv[0] = PortState::Blocked;
                    recv[1] = PortState::Blocked;
                }
            },
            AsOfJoinState::FlushInputBuffer => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
                if recv1_blocked {
                    send[0] = PortState::Blocked;
                }
                if send_blocked {
                    recv[1] = PortState::Blocked;
                }
            },
            AsOfJoinState::Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
                self.left_buffer = VecDeque::new();
                self.in_progress = None;
                self.right_buffer = self.right_buffer.empty_clone();
            },
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        match &self.state {
            AsOfJoinState::Running | AsOfJoinState::FlushInputBuffer => {
                let params = &self.params;
                let recv_left = match self.state {
                    AsOfJoinState::Running => Some(recv_ports[0].take().unwrap().serial()),
                    _ => None,
                };
                let recv_right = recv_ports[1].take().map(RecvPort::serial);
                let send = send_ports[0].take().unwrap().parallel();
                let (distributor, dist_recv) =
                    dc::distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
                let left_buffer = &mut self.left_buffer;
                let in_progress = &mut self.in_progress;
                let right_buffer = &mut self.right_buffer;
                let output_seq = &mut self.output_seq;
                let last_non_null_row_left = &mut self.last_non_null_row_left;
                let unchecked_right_rows = &mut self.unchecked_right_rows;
                let mut right_checks = RightChecks {
                    params,
                    last_row: &mut self.last_non_null_row_right,
                    pending: Vec::new(),
                    pending_rows: 0,
                    batch_rows: get_ideal_morsel_size(),
                    // The rows of earlier phases were all checked when those ended.
                    checked: watch::Sender::new(Some(self.checked_right_rows)),
                    added: &mut self.checked_right_rows,
                    running: VecDeque::new(),
                    // Per pipeline, one batch being checked and one waiting.
                    max_running: 2 * send.len(),
                    task_metrics: state.task_metrics(),
                };
                let checked = right_checks.checked.subscribe();
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    distribute_work_task(
                        recv_left,
                        recv_right,
                        distributor,
                        left_buffer,
                        in_progress,
                        right_buffer,
                        output_seq,
                        last_non_null_row_left,
                        unchecked_right_rows,
                        &mut right_checks,
                        params,
                    )
                    .await?;
                    right_checks.finish().await
                }));

                join_handles.extend(dist_recv.into_iter().zip(send).map(|(recv, send)| {
                    let checked = checked.clone();
                    scope.spawn_task(TaskPriority::High, async move {
                        compute_and_emit_task(recv, send, checked, params).await
                    })
                }));
            },
            AsOfJoinState::Done => {
                unreachable!();
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn distribute_work_task(
    mut recv_left: Option<PortReceiver>,
    mut recv_right: Option<PortReceiver>,
    mut distributor: dc::Sender<Piece>,
    left_buffer: &mut VecDeque<DataFrame>,
    in_progress: &mut Option<(Arc<MorselOutput>, usize)>,
    right_buffer: &mut SpillFrameSearchBuffer,
    output_seq: &mut MorselSeq,
    last_non_null_row_left: &mut Option<DataFrame>,
    unchecked_right_rows: &mut usize,
    right_checks: &mut RightChecks<'_>,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let source_token = SourceToken::new();
    let right_done = recv_right.is_none();
    let check_sortedness = params.as_of_options().check_sortedness;
    // A left morsel is dispatched in pieces once the right buffer holds a morsel's worth of
    // rows, so the buffer does not have to grow to cover all of it.
    let cut_rows = right_checks.batch_rows;

    loop {
        if source_token.stop_requested() {
            for sf in stop_and_take_pipe_contents(recv_left.as_mut()).await {
                left_buffer.push_back(sf.into_df().await);
            }
            let height = right_buffer.height();
            right_buffer
                .stop_and_buffer_from_pipe(recv_right.as_mut())
                .await;
            *unchecked_right_rows += right_buffer.height() - height;
            return Ok(());
        }

        let (output, mut offset, st) = if let Some((output, offset)) = in_progress.take() {
            (output, offset, source_token.clone())
        } else {
            let (left_df, st) = if let Some(df) = left_buffer.pop_front() {
                (df, source_token.clone())
            } else if let Some(ref mut recv) = recv_left
                && let Ok(m) = recv.recv().await
            {
                let (sf, _, st, _) = m.into_inner();
                (sf.into_df().await, st)
            } else {
                let height = right_buffer.height();
                right_buffer
                    .stop_and_buffer_from_pipe(recv_right.as_mut())
                    .await;
                *unchecked_right_rows += right_buffer.height() - height;
                return Ok(());
            };
            if check_sortedness {
                check_left_continuity(last_non_null_row_left, &left_df, params)?;
            }
            let output = MorselOutput {
                left: left_df,
                seq: *output_seq,
                pieces: Mutex::default(),
            };
            *output_seq = output_seq.successor();
            (Arc::new(output), 0, st)
        };
        let left_df = output.left.clone();
        let mut output = Some(output);

        // Checked only now, as a phase that never resumes does not use these rows.
        let n = std::mem::take(unchecked_right_rows);
        if check_sortedness && n > 0 {
            let rows = right_buffer.clone().slice(right_buffer.height() - n, n);
            right_checks.add(&rows.into_df().await).await?;
        }

        // Dispatch the left morsel in pieces: each one once the right rows it can match have
        // all arrived, with only those right rows, and the buffer pruned to what is still
        // needed in between.
        while offset < left_df.height() {
            let rest = left_df.slice(offset as i64, left_df.height() - offset);
            prune_right_side(&rest, right_buffer, params).await?;
            // The length of the next piece and the end of the right rows it can match.
            let (len, end) = loop {
                if right_done {
                    // No more right rows: past the buffer, the matches run to its end.
                    break (
                        rest.height(),
                        right_range_end(&rest, right_buffer, params).await?,
                    );
                }
                if let Some(end) = right_range_end(&rest, right_buffer, params).await? {
                    break (rest.height(), Some(end));
                }
                if right_buffer.height() >= cut_rows {
                    let len = covered_rows(&rest, right_buffer, params).await?;
                    if len > 0
                        && let Some(end) =
                            right_range_end(&rest.slice(0, len), right_buffer, params).await?
                    {
                        break (len, Some(end));
                    }
                }
                if let Some(ref mut recv) = recv_right
                    && let Ok(morsel_right) = recv.recv().await
                {
                    let sf = morsel_right.into_sf();
                    if check_sortedness {
                        right_checks.add(&*sf.get().await).await?;
                    }
                    right_buffer.push_sf(sf).await;
                    if right_buffer.height() >= cut_rows {
                        prune_right_side(&rest, right_buffer, params).await?;
                    }
                } else {
                    // The right pipe is empty at this stage, we will need to wait for
                    // a new stage and try again.
                    *in_progress = Some((output.unwrap(), offset));
                    for sf in stop_and_take_pipe_contents(recv_left.as_mut()).await {
                        left_buffer.push_back(sf.into_df().await);
                    }
                    return Ok(());
                }
            };

            let last = offset + len == left_df.height();
            if check_sortedness && last {
                right_checks.flush().await?;
            }
            // The searches above may have read any right row received, so the morsel is sent
            // only once all of them are known to be sorted.
            let piece = Piece {
                left: rest.slice(0, len),
                right: right_buffer.clone().slice(0, end.unwrap_or(usize::MAX)),
                offset,
                right_rows: *right_checks.added,
                // The worker joining the last piece sends the morsel, so the last piece takes
                // the only other handle to it.
                output: if last { output.take() } else { output.clone() }.unwrap(),
                source_token: st.clone(),
            };
            if distributor.send(piece).await.is_err() {
                return Ok(());
            }
            offset += len;
        }
    }
}

/// A left morsel, joined in pieces and sent as one morsel once all are joined.
#[derive(Debug)]
struct MorselOutput {
    left: DataFrame,
    seq: MorselSeq,
    /// The joined pieces, with their offset in the morsel and their `right_rows`.
    pieces: Mutex<Vec<(usize, DataFrame, usize)>>,
}

/// A piece of a left morsel, with the right rows it can match.
struct Piece {
    left: DataFrame,
    right: SpillFrameSearchBuffer,
    offset: usize,
    /// The number of right rows received when the piece was cut.
    right_rows: usize,
    output: Arc<MorselOutput>,
    source_token: SourceToken,
}

/// Checks that the right rows are sorted, in batches, in tasks that run alongside the join.
/// Every right row is checked, also those pruned before any worker sees them.
struct RightChecks<'a> {
    params: &'a Arc<AsOfJoinParams>,
    /// The last row checked, to check the order across the boundary to the next rows.
    last_row: &'a mut Option<DataFrame>,
    /// The sort columns of the rows not checked yet.
    pending: Vec<DataFrame>,
    pending_rows: usize,
    /// The rows are checked in batches of a morsel's worth.
    batch_rows: usize,
    added: &'a mut usize,
    /// The number of rows known to be sorted, or `None` once a check failed.
    checked: watch::Sender<Option<usize>>,
    running: VecDeque<AbortOnDropHandle<PolarsResult<()>>>,
    max_running: usize,
    task_metrics: Option<&'a TaskMetricAggregator>,
}

impl RightChecks<'_> {
    async fn add(&mut self, df: &DataFrame) -> PolarsResult<()> {
        if df.height() == 0 {
            return Ok(());
        }
        let params = self.params;
        let cols = params.right_by().iter().chain([params.right.key_col()]);
        self.pending.push(df.select(cols.cloned())?);
        self.pending_rows += df.height();
        *self.added += df.height();
        if self.pending_rows >= self.batch_rows {
            self.flush().await?;
        }
        Ok(())
    }

    /// Starts checking the pending rows, together with the last row checked before them.
    async fn flush(&mut self) -> PolarsResult<()> {
        let Some(last) = self.pending.last() else {
            return Ok(());
        };
        let prev = self.last_row.replace(last.slice(-1, 1));
        let frames = std::mem::take(&mut self.pending);
        let (start, end) = (*self.added - self.pending_rows, *self.added);
        self.pending_rows = 0;
        while self.running.len() >= self.max_running {
            self.running.pop_front().unwrap().await?;
        }
        let params = self.params.clone();
        let checked = self.checked.clone();
        let check_rows = move || {
            let (by, key) = (params.right_by(), params.right.key_col());
            if let [rows] = &frames[..] {
                // Check the boundary on its own: with `prev` in front, the comparison kernels
                // would rechunk the frame.
                let cols = by.iter().chain([key]);
                let descending = params.by_descending.iter().chain([&false]);
                let nulls_last = params.by_nulls_last.iter().chain([&false]);
                check_continuity(prev, Some(rows.slice(0, 1)), cols, descending, nulls_last)?;
                return check_df_sorted(rows, by, key, &params);
            }
            let rows = accumulate_dataframes_vertical_unchecked(prev.into_iter().chain(frames));
            check_df_sorted(&rows, by, key, &params)
        };
        let check = async move {
            let result = check_rows();
            // The rows are known to be sorted once the rows before them are.
            let _ = checked
                .subscribe()
                .wait_for(|c| c.is_none_or(|c| c >= start))
                .await;
            checked.send_modify(|c| *c = c.and(result.is_ok().then_some(end)));
            result
        };
        let task = executor::spawn(TaskPriority::High, self.task_metrics, check);
        self.running.push_back(AbortOnDropHandle::new(task));
        Ok(())
    }

    async fn finish(mut self) -> PolarsResult<()> {
        self.flush().await?;
        for check in self.running {
            check.await?;
        }
        Ok(())
    }
}

/// The number of leading rows of `left` whose matches the buffered right rows determine: those
/// before the last right row, in its group before its key, or for `Nearest`, before the run of
/// equal keys it ends.
async fn covered_rows(
    left: &DataFrame,
    right: &SpillFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<usize> {
    let options = params.as_of_options();
    let last = right.height() - 1;
    let mut group = Vec::with_capacity(params.right_by().len());
    for col in params.right_by() {
        // SAFETY: `last` is in bounds.
        group.push(unsafe { right.get_unchecked(col, last).await });
    }
    let key_col = params.right.key_col();
    // SAFETY: `last` is in bounds.
    let mut bound = Some(unsafe { right.get_unchecked(key_col, last).await });
    if options.strategy == Nearest {
        let (start, _) = right_group_range(right, &group, params).await;
        let key = bound.take().unwrap();
        let run_start = right
            .binary_search(|x| *x >= key, key_col, start..last)
            .await;
        if run_start > start {
            // SAFETY: `run_start - 1` is in bounds.
            bound = Some(unsafe { right.get_unchecked(key_col, run_start - 1).await });
        }
    }
    let inclusive = matches!(
        (options.strategy, options.allow_eq),
        (Forward, true) | (Backward, false)
    );

    let left_by: Vec<_> = params
        .left_by()
        .iter()
        .map(|c| left.column(c))
        .try_collect()?;
    let left_key = left.column(params.left.key_col())?;
    let is_covered = |i| -> PolarsResult<bool> {
        let reorder = Iterator::zip(params.by_descending.iter(), params.by_nulls_last.iter());
        for ((col, value), (descending, nulls_last)) in left_by.iter().zip(&group).zip(reorder) {
            match reorder_cmp(&col.get(i)?, value, *descending, *nulls_last) {
                Ordering::Equal => {},
                ord => return Ok(ord.is_lt()),
            }
        }
        let key = left_key.get(i)?;
        Ok(key.is_null()
            || bound
                .as_ref()
                .is_some_and(|b| key < *b || (inclusive && key == *b)))
    };
    let (mut lo, mut hi) = (0, left.height());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if is_covered(mid)? {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// The range of the right rows in the group with the `by` values `group`.
async fn right_group_range(
    right: &SpillFrameSearchBuffer,
    group: &[AnyValue<'_>],
    params: &AsOfJoinParams,
) -> (usize, usize) {
    let (mut start, mut end) = (0, right.height());
    let reorder = Iterator::zip(params.by_descending.iter(), params.by_nulls_last.iter());
    for ((right_by, value), (descending, nulls_last)) in
        params.right_by().iter().zip(group).zip(reorder)
    {
        let cmp = |x: &AnyValue<'_>| reorder_cmp(x, value, *descending, *nulls_last);
        start = right
            .binary_search(|x| cmp(x).is_ge(), right_by, start..end)
            .await;
        end = right
            .binary_search(|x| cmp(x).is_gt(), right_by, start..end)
            .await;
    }
    (start, end)
}

/// The `by` values of row `i` of `left`.
fn left_group(
    left: &DataFrame,
    i: usize,
    params: &AsOfJoinParams,
) -> PolarsResult<Vec<AnyValue<'static>>> {
    params
        .left_by()
        .iter()
        .map(|c| Ok(left.column(c)?.get(i)?.into_static()))
        .collect()
}

/// Check that the first row of the DataFrame is in order with respect to the value in prev_row.
fn check_left_continuity(
    last_non_null_row: &mut Option<DataFrame>,
    df: &DataFrame,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    if df.height() == 0 {
        return Ok(());
    }

    let key_col_name = params.left.key_col();
    let sorted_by_cols = params.left_by().iter().chain([key_col_name]);
    let sorted_by_descending = params.by_descending.iter().chain([&false]);
    let sorted_by_nulls_last = params.by_nulls_last.iter().chain([&false]);
    let project = df.select(sorted_by_cols.clone())?;
    let first_non_null_at = project.column(key_col_name)?.first_non_null();
    let first_non_null_row = first_non_null_at.map(|at| project.slice(at as i64, 1));
    check_continuity(
        last_non_null_row.clone(),
        first_non_null_row,
        sorted_by_cols,
        sorted_by_descending,
        sorted_by_nulls_last,
    )?;

    // Store the last non-null row of this DataFrame for the next check.
    if let Some(pos) = project.column(key_col_name)?.last_non_null() {
        *last_non_null_row = Some(project.slice((pos) as i64, 1));
    }
    Ok(())
}

fn check_continuity<'a, IS, IB>(
    before_split_point_row: Option<DataFrame>,
    after_split_point_row: Option<DataFrame>,
    sorted_by_cols: IS,
    sorted_by_descending: IB,
    sorted_by_nulls_last: IB,
) -> PolarsResult<()>
where
    IS: Iterator<Item = &'a PlSmallStr> + Clone,
    IB: Iterator<Item = &'a bool> + Clone,
{
    if let Some(before_split_point) = before_split_point_row
        && let Some(after_split_point) = after_split_point_row
    {
        let across_split_point = before_split_point.vstack(&after_split_point)?;
        if !across_split_point.is_sorted(
            &sorted_by_cols.cloned().collect_vec(),
            &sorted_by_descending.cloned().collect_vec(),
            &sorted_by_nulls_last.cloned().collect_vec(),
        )? {
            return Err(not_sorted_err());
        }
    }
    Ok(())
}

/// The end of the right rows the last row of `left` with a non-null key can match, or
/// `None` if more right rows are needed to know it.
async fn right_range_end(
    left: &DataFrame,
    right: &SpillFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<Option<usize>> {
    // Rows with a null key match nothing.
    let Some(last) = left.column(params.left.key_col())?.last_non_null() else {
        return Ok(Some(0));
    };
    if right.height() == 0 {
        return Ok(None);
    }

    let options = params.as_of_options();

    let group = left_group(left, last, params)?;
    let (start, end) = right_group_range(right, &group, params).await;
    if start >= right.height() {
        return Ok(None);
    }
    // Past the end of the group in the buffer, the group is complete if more rows follow.
    let group_end = (end < right.height()).then_some(end);

    let left_key = left.column(params.left.key_col())?.as_materialized_series();
    // SAFETY: `last` is in bounds.
    let left_last_val = unsafe { left_key.get_unchecked(last) };
    let match_end = match (options.strategy, options.allow_eq) {
        (Forward, true) | (Backward, false) => {
            right
                .binary_search(|x| *x >= left_last_val, params.right.key_col(), start..end)
                .await
        },
        (Forward, false) | (Backward, true) => {
            right
                .binary_search(|x| *x > left_last_val, params.right.key_col(), start..end)
                .await
        },
        (Nearest, _) => {
            let first_greater = right
                .binary_search(|x| *x > left_last_val, params.right.key_col(), start..end)
                .await;
            if first_greater >= end {
                return Ok(group_end);
            }
            // In the Nearest case, there may be a chunk of consecutive equal
            // values following the match value on the left side.  In this case,
            // the AsOf join is greedy and should until the *end* of that chunk.

            // SAFETY: We just checked that first_greater is in bounds
            let first_greater_val = unsafe {
                right
                    .get_unchecked(params.right.key_col(), first_greater)
                    .await
            };
            right
                .binary_search(
                    |x| *x > first_greater_val,
                    params.right.key_col(),
                    first_greater..end,
                )
                .await
        },
    };
    if match_end >= end {
        return Ok(group_end);
    }
    // A forward match is the row found itself.
    Ok(Some(match_end + usize::from(options.strategy == Forward)))
}

/// Prune right-side rows that are no longer needed using the first row of `left` with a
/// non-null key as the pruning reference point.
async fn prune_right_side(
    left: &DataFrame,
    right: &mut SpillFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let Some(left_row_idx) = left.column(params.left.key_col())?.first_non_null() else {
        return Ok(());
    };
    if right.height() == 0 {
        return Ok(());
    }

    let group = left_group(left, left_row_idx, params)?;
    let (start, end) = right_group_range(right, &group, params).await;

    let left_key = left.column(params.left.key_col())?.as_materialized_series();
    // SAFETY: `left_row_idx` is in bounds.
    let key_val = unsafe { left_key.get_unchecked(left_row_idx) };
    let mut right_range_start = right
        .binary_search(|x| *x >= key_val, params.right.key_col(), start..end)
        .await;
    if matches!(params.as_of_options().strategy, Backward | Nearest) {
        right_range_start = right_range_start.saturating_sub(1).max(start);
    }

    right.drop_front(right_range_start);
    Ok(())
}

async fn compute_and_emit_task(
    mut dist_recv: dc::Receiver<Piece>,
    mut send: PortSender,
    mut right_checked: watch::Receiver<Option<usize>>,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let wait_group = WaitGroup::default();
    let mut scratch1 = ScratchVec::default();
    let mut scratch2 = ScratchVec::default();
    while let Ok(piece) = dist_recv.recv().await {
        let out = compute_asof_join(
            piece.left,
            piece.right,
            params,
            &mut scratch1,
            &mut scratch2,
        )
        .await?;
        piece.output.pieces.lock().push((piece.offset, out, piece.right_rows));
        // The worker joining the last piece of a morsel sends it.
        let Some(output) = Arc::into_inner(piece.output) else {
            continue;
        };
        let mut pieces = output.pieces.into_inner();
        pieces.sort_unstable_by_key(|(offset, ..)| *offset);
        if params.as_of_options().check_sortedness {
            check_df_sorted(
                &output.left,
                params.left_by(),
                params.left.key_col(),
                params,
            )?;
        }
        let right_rows = pieces.iter().map(|(.., rows)| *rows).max().unwrap_or(0);
        let checked = right_checked.wait_for(|c| c.is_none_or(|c| c >= right_rows));
        match checked.await.map(|c| *c) {
            Ok(Some(_)) => {},
            Ok(None) => return Err(not_sorted_err()),
            // The distributor stopped with an error.
            Err(_) => return Ok(()),
        }
        let out = accumulate_dataframes_vertical_unchecked(pieces.into_iter().map(|(_, df, _)| df));
        let mut morsel = Morsel::new_unregistered(out, output.seq, piece.source_token);
        morsel.set_consume_token(wait_group.token());
        if send.send(morsel).await.is_err() {
            return Ok(());
        }
        wait_group.wait().await;
    }
    Ok(())
}

/// Encodes the "by" group columns of a DataFrame for sorted group traversal.
///
/// Provides run-length group boundaries and value comparison, abstracting over
/// the single-column and multi-column cases.
struct ByGroups<'a> {
    kind: GroupEncodingKind,
    /// Run-lengths of consecutive equal group values (in sorted order).
    run_lengths: &'a mut Vec<IdxSize>,
}

enum GroupEncodingKind {
    /// Single by-column stored directly.
    Single(Column),
    /// Multiple by-columns merged into a row-encoded binary column for comparison.
    Multi(ChunkedArray<BinaryOffsetType>),
}

impl<'a> ByGroups<'a> {
    fn find_groups(
        df: &DataFrame,
        by: &[PlSmallStr],
        run_lengths: &'a mut ScratchVec<IdxSize>,
        params: &AsOfJoinParams,
    ) -> PolarsResult<Self> {
        let run_lengths = run_lengths.get();
        let kind = match by {
            [col_name] => {
                let col = df.column(col_name)?;
                rle_lengths(col, run_lengths)?;
                GroupEncodingKind::Single(col.clone())
            },
            _ => {
                let cols = by
                    .iter()
                    .map(|n| df.column(n).unwrap().clone())
                    .collect_vec();
                let encoded = _get_rows_encoded_ca(
                    ROW_ENCODED_COL_NAME,
                    &cols,
                    &params.by_descending,
                    &params.by_nulls_last,
                    true,
                )?;
                rle_lengths_helper_ca(&encoded, run_lengths);
                GroupEncodingKind::Multi(encoded)
            },
        };
        Ok(Self { kind, run_lengths })
    }

    /// Returns true if the group value at `idx` is null.
    ///
    /// # Safety
    /// `idx` must be a valid row index.
    unsafe fn is_null_at(&self, idx: usize) -> bool {
        match &self.kind {
            GroupEncodingKind::Single(col) => unsafe { col.get_unchecked(idx) }.is_null(),
            GroupEncodingKind::Multi(enc) => unsafe { enc.get_unchecked(idx) }.is_none(),
        }
    }

    /// Lexicographically compare group `idx` with group `other_idx`.
    ///
    /// # Safety
    /// Both indices must be valid row indices within their respective encodings.
    unsafe fn cmp_at(
        &self,
        self_idx: usize,
        other: &Self,
        other_idx: usize,
        params: &AsOfJoinParams,
    ) -> Ordering {
        match (&self.kind, &other.kind) {
            (GroupEncodingKind::Single(s), GroupEncodingKind::Single(o)) => reorder_cmp(
                &unsafe { s.get_unchecked(self_idx) },
                &unsafe { o.get_unchecked(other_idx) },
                params.by_descending[0],
                params.by_nulls_last[0],
            ),
            (GroupEncodingKind::Multi(s), GroupEncodingKind::Multi(o)) => reorder_cmp(
                &unsafe { s.get_unchecked(self_idx) },
                &unsafe { o.get_unchecked(other_idx) },
                false,
                false,
            ),
            _ => unreachable!("mismatched GroupEncoding kinds"),
        }
    }

    /// Iterates over groups as `(start_row, row_count)` pairs.
    fn iter_groups(&self) -> impl Iterator<Item = (usize, usize)> {
        self.run_lengths.iter().scan(0usize, |offset, &len| {
            let start = *offset;
            *offset += len as usize;
            Some((start, len as usize))
        })
    }
}

fn drop_columns(
    left_df: &mut DataFrame,
    right_df: &mut DataFrame,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    // Drop any temporary join-key columns that were added before calling _join_asof.
    if let Some(ref col) = params.left.tmp_key_col
        && left_df.schema().contains(col)
    {
        left_df.drop_in_place(col)?;
    }
    if let Some(ref col) = params.right.tmp_key_col
        && right_df.schema().contains(col)
    {
        right_df.drop_in_place(col)?;
    }

    // Only return one set of group columns in the result.
    for col in params.right_by() {
        right_df.drop_in_place(col)?;
    }

    // Coalesce the key column.
    if params.args.should_coalesce() && params.left.on == params.right.on {
        right_df.drop_in_place(&params.right.on)?;
    }
    Ok(())
}

/// Calls `_join_asof` on a left/right pair and cleans up the output via
/// [`finalize_output`].
fn join_asof_ungrouped(
    mut left_df: DataFrame,
    mut right_df: DataFrame,
    left_key: &Series,
    right_key: &Series,
    options: &AsOfOptions,
    params: &AsOfJoinParams,
) -> PolarsResult<DataFrame> {
    _check_asof_columns(
        left_key,
        right_key,
        options.tolerance.is_some(),
        false, // Sortedness is already checked
        false,
    )?;

    let take_idx = _join_asof_dispatch(
        left_key,
        right_key,
        options.strategy,
        options.tolerance.clone().map(Scalar::into_value),
        options.allow_eq,
    )?;
    drop_columns(&mut left_df, &mut right_df, params)?;
    // SAFETY: _join_asof_dispatch only returns in-bounds indices.
    let right_df = unsafe { right_df.take_unchecked(&take_idx) };
    _finish_join(left_df, right_df, params.args.suffix.clone())
}

async fn compute_asof_join(
    mut left_df: DataFrame,
    right_dfsb: SpillFrameSearchBuffer,
    params: &AsOfJoinParams,
    left_lengths: &mut ScratchVec<IdxSize>,
    right_lengths: &mut ScratchVec<IdxSize>,
) -> PolarsResult<DataFrame> {
    let mut right_df = right_dfsb.into_df().await;
    let options = params.as_of_options();
    let left_key = left_df.column(params.left.key_col())?.to_physical_repr();
    let right_key = right_df
        .column(params.right.key_col())?
        .clone()
        .to_physical_repr();

    if params.left_by().is_empty() {
        return join_asof_ungrouped(
            left_df,
            right_df,
            left_key.as_materialized_series(),
            right_key.as_materialized_series(),
            options,
            params,
        );
    }

    let left_groups = ByGroups::find_groups(&left_df, params.left_by(), left_lengths, params)?;
    let right_groups = ByGroups::find_groups(&right_df, params.right_by(), right_lengths, params)?;
    let cmp_at = |left_idx, right_idx| unsafe {
        ByGroups::cmp_at(&left_groups, left_idx, &right_groups, right_idx, params)
    };
    let left_groups_iter = left_groups.iter_groups();
    let mut right_groups_iter = right_groups.iter_groups();
    let mut right_group = right_groups_iter.next();

    drop_columns(&mut left_df, &mut right_df, params)?;

    // The right row each left row matches, gathered once for all groups.
    let mut take = Vec::with_capacity(left_df.height());
    for (left_start, left_group_len) in left_groups_iter {
        let right_chunk_len =
            // SAFETY: the iterator results will be in-bounds.
            if unsafe { left_groups.is_null_at(left_start) }  {
                // If the left group is null, then it will have no matches on the right side.
                0
            } else {
                // Advance the right cursor past null groups and any groups that are
                // ordered before the left group.
                let should_bump_right_group =  |right_start| unsafe {
                    right_groups.is_null_at(right_start) || cmp_at(left_start, right_start).is_gt()
                };
                while let Some((right_start, _)) = right_group
                    && should_bump_right_group(right_start)
                {
                    right_group = right_groups_iter.next();
                }
                if let Some((right_start, right_len)) = right_group
                    && cmp_at(left_start, right_start).is_eq()
                {
                    right_len
                } else {
                    // The left group is not present in the right group.
                    0
                }
            };

        let right_start = right_group.map_or(0, |(start, _len)| start);
        let group_left_key = left_key.slice(left_start as i64, left_group_len);
        let group_right_key = right_key.slice(right_start as i64, right_chunk_len);

        let take_idx = _join_asof_dispatch(
            group_left_key.as_materialized_series(),
            group_right_key.as_materialized_series(),
            options.strategy,
            options.tolerance.clone().map(Scalar::into_value),
            options.allow_eq,
        )?;

        take.extend(
            take_idx
                .iter()
                .map(|i| i.map(|i| i + right_start as IdxSize)),
        );
    }

    let take = IdxCa::from_iter_options(PlSmallStr::EMPTY, take.into_iter());
    // SAFETY: each index is in bounds of the group's slice of `right_df`, offset by its start.
    let out_right = unsafe { right_df.take_unchecked(&take) };
    _finish_join(left_df, out_right, params.args.suffix.clone())
}

fn check_df_sorted(
    dataframe: &DataFrame,
    by: &[PlSmallStr],
    on: &PlSmallStr,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let sorted_by_cols = by.iter().chain([on]);
    let sorted_by_descending = params.by_descending.iter().chain([&false]);
    let sorted_by_nulls_last = params.by_nulls_last.iter().chain([&false]);
    if !dataframe.is_sorted(
        &sorted_by_cols.cloned().collect_vec(),
        &sorted_by_descending.cloned().collect_vec(),
        &sorted_by_nulls_last.cloned().collect_vec(),
    )? {
        return Err(not_sorted_err());
    }
    Ok(())
}

fn not_sorted_err() -> PolarsError {
    polars_err!(InvalidOperation: "argument in operation 'asof_join' is not sorted, please sort the 'expr/series/column' first")
}
