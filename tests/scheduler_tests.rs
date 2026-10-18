use minisgl::core::req::{BatchPhase, Req};
use minisgl::core::sampling::SamplingParams;
use minisgl::scheduler::scheduler::Scheduler;

#[test]
fn test_scheduler_empty() {
    let scheduler = Scheduler::new(10);
    let mut reqs = vec![];
    let batch = scheduler.schedule(&mut reqs);
    assert!(batch.is_none());
}

#[test]
fn test_scheduler_active_requests() {
    let scheduler = Scheduler::new(10);
    let params = SamplingParams::default();
    let req1 = Req::new(1, vec![100, 200], 0, 0, 5, params.clone());
    let req2 = Req::new(2, vec![300, 400], 0, 0, 5, params);

    let mut reqs = vec![req1, req2];
    let batch = scheduler.schedule(&mut reqs);
    assert!(batch.is_some());

    let batch = batch.unwrap();
    assert_eq!(batch.size(), 2);
    assert_eq!(batch.phase, BatchPhase::Decode);
}
