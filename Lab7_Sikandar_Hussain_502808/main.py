import argparse
from pathlib import Path

from tester import APIPerformanceTester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 7: API stress testing, latency analysis, and robustness checks"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="API base URL")
    parser.add_argument("--endpoint", default="/predict", help="Prediction endpoint path")
    parser.add_argument("--api-key", default="", help="Optional API key for X-API-Key header")
    parser.add_argument("--timeout", type=float, default=5.0, help="Request timeout in seconds")
    parser.add_argument("--latency-requests", type=int, default=50, help="Requests for latency test")
    parser.add_argument("--throughput-requests", type=int, default=150, help="Requests for throughput test")
    parser.add_argument("--concurrent-users", type=int, default=50, help="Users for main stress test")
    parser.add_argument("--requests-per-user", type=int, default=4, help="Requests per concurrent user")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    headers = {}
    if args.api_key.strip():
        headers["X-API-Key"] = args.api_key.strip()

    tester = APIPerformanceTester(
        base_url=args.base_url,
        endpoint=args.endpoint,
        headers=headers,
        timeout=args.timeout,
    )

    valid_payload = {
        "pclass": 3,
        "sex": "male",
        "age": 22,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
    }

    edge_cases = [
        ("missing_fields", {}),
        (
            "wrong_data_type",
            {
                "pclass": "third",
                "sex": "male",
                "age": "young",
                "sibsp": 1,
                "parch": 0,
                "fare": "cheap",
                "embarked": "S",
            },
        ),
        (
            "extreme_values",
            {
                "pclass": 1,
                "sex": "female",
                "age": 9999,
                "sibsp": 500,
                "parch": 500,
                "fare": 999999,
                "embarked": "C",
            },
        ),
        (
            "incorrect_shape_missing_field",
            {
                "pclass": 3,
                "sex": "male",
                "age": 30,
                "sibsp": 0,
                "parch": 0,
                "fare": 7.9,
            },
        ),
    ]

    latency = tester.measure_latency(valid_payload, num_requests=args.latency_requests)
    throughput = tester.measure_throughput(valid_payload, num_requests=args.throughput_requests)
    stress = tester.stress_test(
        valid_payload,
        concurrent_users=args.concurrent_users,
        requests_per_user=args.requests_per_user,
    )

    breaking_point = tester.find_breaking_point(
        valid_payload,
        user_steps=[1, 5, 10, 20, 40, 60, 80],
        requests_per_user=2,
    )
    robustness = tester.analyze_robustness(breaking_point)
    edge_case_results = tester.test_edge_cases(edge_cases)

    report_text = tester.render_report(
        api_url=f"{args.base_url.rstrip('/')}{args.endpoint}",
        latency=latency,
        throughput=throughput,
        stress=stress,
        edge_cases=edge_case_results,
        breaking_point=breaking_point,
        robustness=robustness,
    )

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("Lab 7 testing completed.")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
