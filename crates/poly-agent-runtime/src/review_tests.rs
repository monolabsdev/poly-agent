use poly_agent_core::{AutoReviewRisk, ReviewDecision};

use super::{parse_review_response, ReviewVerdict};

#[test]
fn parses_well_formed_response() {
    let raw = r#"{"risk":"low","decision":"approve","reason":"Read-only."}"#;
    let verdict = parse_review_response(raw);
    assert_eq!(verdict.risk, AutoReviewRisk::Low);
    assert_eq!(verdict.decision, ReviewDecision::Approve);
    assert_eq!(verdict.reason, "Read-only.");
}

#[test]
fn parses_response_with_prose_around_json() {
    let raw = "Sure, here is the verdict:\n{\"risk\":\"high\",\"decision\":\"deny\",\"reason\":\"deletes files\"}\nDone.";
    let verdict = parse_review_response(raw);
    assert_eq!(verdict.risk, AutoReviewRisk::High);
    assert_eq!(verdict.decision, ReviewDecision::Deny);
    assert_eq!(verdict.reason, "deletes files");
}

#[test]
fn invalid_json_falls_back_to_medium_ask() {
    let verdict = parse_review_response("not json at all");
    assert_eq!(
        verdict,
        ReviewVerdict {
            risk: AutoReviewRisk::Medium,
            decision: ReviewDecision::Ask,
            reason: "Auto-review failed to return a valid structured decision.".to_string(),
        }
    );
}

#[test]
fn missing_fields_fall_back_safely() {
    let raw = r#"{"risk":"low"}"#;
    let verdict = parse_review_response(raw);
    assert_eq!(verdict.risk, AutoReviewRisk::Medium);
    assert_eq!(verdict.decision, ReviewDecision::Ask);
}

#[test]
fn unknown_risk_falls_back_safely() {
    let raw = r#"{"risk":"extreme","decision":"approve","reason":"x"}"#;
    let verdict = parse_review_response(raw);
    assert_eq!(verdict.risk, AutoReviewRisk::Medium);
    assert_eq!(verdict.decision, ReviewDecision::Ask);
}

#[test]
fn unknown_decision_falls_back_safely() {
    let raw = r#"{"risk":"low","decision":"maybe","reason":"x"}"#;
    let verdict = parse_review_response(raw);
    assert_eq!(verdict.risk, AutoReviewRisk::Medium);
    assert_eq!(verdict.decision, ReviewDecision::Ask);
}

#[test]
fn empty_reason_is_replaced_with_default() {
    let raw = r#"{"risk":"low","decision":"approve","reason":"   "}"#;
    let verdict = parse_review_response(raw);
    assert_eq!(verdict.risk, AutoReviewRisk::Low);
    assert_eq!(verdict.decision, ReviewDecision::Approve);
    assert!(!verdict.reason.is_empty());
}
