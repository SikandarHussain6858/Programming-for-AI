import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class RequestResult:
    success: bool
    status_code: Optional[int]
    latency_ms: float
    error: Optional[str] = None


class APIPerformanceTester:
    def __init__(
        self,
        base_url: str,
        endpoint: str = "/predict",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.url = f"{self.base_url}{self.endpoint}"
        self.headers = headers or {}
        self.timeout = timeout

    def send_request(self, payload: Dict[str, Any]) -> RequestResult:
        start = time.perf_counter()
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                success=200 <= response.status_code < 300,
                status_code=response.status_code,
                latency_ms=latency_ms,
                error=None,
            )
        except requests.RequestException as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                success=False,
                status_code=None,
                latency_ms=latency_ms,
                error=str(exc),
            )

    def send_invalid_json(self, raw_data: str) -> RequestResult:
        start = time.perf_counter()
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        try:
            response = requests.post(
                self.url,
                data=raw_data,
                headers=headers,
                timeout=self.timeout,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                success=200 <= response.status_code < 300,
                status_code=response.status_code,
                latency_ms=latency_ms,
                error=None,
            )
        except requests.RequestException as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                success=False,
                status_code=None,
                latency_ms=latency_ms,
                error=str(exc),
            )

    def measure_latency(self, payload: Dict[str, Any], num_requests: int) -> Dict[str, Any]:
        results = [self.send_request(payload) for _ in range(num_requests)]
        latencies = [r.latency_ms for r in results]
        success_count = sum(1 for r in results if r.success)

        return {
            "count": num_requests,
            "success_count": success_count,
            "failure_count": num_requests - success_count,
            "avg_ms": statistics.mean(latencies) if latencies else 0.0,
            "min_ms": min(latencies) if latencies else 0.0,
            "max_ms": max(latencies) if latencies else 0.0,
            "p95_ms": self._percentile(latencies, 95),
            "p99_ms": self._percentile(latencies, 99),
        }

    def measure_throughput(self, payload: Dict[str, Any], num_requests: int) -> Dict[str, Any]:
        start = time.perf_counter()
        results = [self.send_request(payload) for _ in range(num_requests)]
        elapsed = time.perf_counter() - start

        success_count = sum(1 for r in results if r.success)
        throughput = num_requests / elapsed if elapsed > 0 else 0.0

        return {
            "count": num_requests,
            "success_count": success_count,
            "failure_count": num_requests - success_count,
            "elapsed_s": elapsed,
            "requests_per_second": throughput,
        }

    def stress_test(
        self,
        payload: Dict[str, Any],
        concurrent_users: int,
        requests_per_user: int,
    ) -> Dict[str, Any]:
        total_requests = concurrent_users * requests_per_user
        results: List[RequestResult] = []

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(self.send_request, payload)
                for _ in range(total_requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.perf_counter() - start

        latencies = [r.latency_ms for r in results]
        success_count = sum(1 for r in results if r.success)

        return {
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "total_requests": total_requests,
            "elapsed_s": elapsed,
            "success_count": success_count,
            "failure_count": total_requests - success_count,
            "success_rate": (success_count / total_requests) * 100 if total_requests else 0.0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "max_latency_ms": max(latencies) if latencies else 0.0,
            "throughput_rps": total_requests / elapsed if elapsed > 0 else 0.0,
        }

    def find_breaking_point(
        self,
        payload: Dict[str, Any],
        user_steps: List[int],
        requests_per_user: int,
    ) -> List[Dict[str, Any]]:
        observations = []
        for users in user_steps:
            stress = self.stress_test(payload, users, requests_per_user)
            observations.append(stress)
        return observations

    def test_edge_cases(
        self,
        edge_cases: List[Tuple[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for name, payload in edge_cases:
            result = self.send_request(payload)
            results.append(
                {
                    "name": name,
                    "status_code": result.status_code,
                    "success": result.success,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                }
            )

        invalid_json_result = self.send_invalid_json("{not-valid-json}")
        results.append(
            {
                "name": "invalid_json_body",
                "status_code": invalid_json_result.status_code,
                "success": invalid_json_result.success,
                "latency_ms": invalid_json_result.latency_ms,
                "error": invalid_json_result.error,
            }
        )
        return results

    @staticmethod
    def analyze_robustness(
        breaking_point_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        slowdown_point = None
        failure_point = None

        for row in breaking_point_results:
            if slowdown_point is None and row["avg_latency_ms"] > 300:
                slowdown_point = row["concurrent_users"]
            if failure_point is None and row["success_rate"] < 95:
                failure_point = row["concurrent_users"]

        if slowdown_point is None and breaking_point_results:
            slowdown_point = "> " + str(breaking_point_results[-1]["concurrent_users"])
        if failure_point is None and breaking_point_results:
            failure_point = "> " + str(breaking_point_results[-1]["concurrent_users"])

        return {
            "slowdown_point_users": slowdown_point,
            "failure_point_users": failure_point,
        }

    @staticmethod
    def render_report(
        api_url: str,
        latency: Dict[str, Any],
        throughput: Dict[str, Any],
        stress: Dict[str, Any],
        edge_cases: List[Dict[str, Any]],
        breaking_point: List[Dict[str, Any]],
        robustness: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        lines.append("Lab 7 - API Stress Testing, Latency Analysis & System Robustness")
        lines.append("=" * 72)
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Target API: {api_url}")
        lines.append("")

        lines.append("1) Valid Request Verification")
        lines.append("- Successful responses during latency run: "
                     f"{latency['success_count']}/{latency['count']}")
        lines.append("")

        lines.append("2) Latency Metrics")
        lines.append(f"- Requests tested: {latency['count']}")
        lines.append(f"- Average latency: {latency['avg_ms']:.2f} ms")
        lines.append(f"- Min latency: {latency['min_ms']:.2f} ms")
        lines.append(f"- Max latency: {latency['max_ms']:.2f} ms")
        lines.append(f"- P95 latency: {latency['p95_ms']:.2f} ms")
        lines.append(f"- P99 latency: {latency['p99_ms']:.2f} ms")
        lines.append("")

        lines.append("3) Throughput Metrics")
        lines.append(f"- Requests tested: {throughput['count']}")
        lines.append(f"- Total elapsed time: {throughput['elapsed_s']:.2f} s")
        lines.append(f"- Throughput: {throughput['requests_per_second']:.2f} req/s")
        lines.append("")

        lines.append("4) Stress Test (Concurrent Users)")
        lines.append(f"- Concurrent users: {stress['concurrent_users']}")
        lines.append(f"- Requests per user: {stress['requests_per_user']}")
        lines.append(f"- Total requests: {stress['total_requests']}")
        lines.append(f"- Success rate: {stress['success_rate']:.2f}%")
        lines.append(f"- Average latency under load: {stress['avg_latency_ms']:.2f} ms")
        lines.append(f"- Max latency under load: {stress['max_latency_ms']:.2f} ms")
        lines.append(f"- Throughput under load: {stress['throughput_rps']:.2f} req/s")
        lines.append("")

        lines.append("5) Failure / Slowdown Point Analysis")
        for row in breaking_point:
            lines.append(
                "- users={users}, success_rate={success:.2f}%, avg_latency={lat:.2f} ms, "
                "throughput={thr:.2f} req/s".format(
                    users=row["concurrent_users"],
                    success=row["success_rate"],
                    lat=row["avg_latency_ms"],
                    thr=row["throughput_rps"],
                )
            )
        lines.append(
            f"- Estimated slowdown point (avg latency > 300 ms): "
            f"{robustness['slowdown_point_users']} users"
        )
        lines.append(
            f"- Estimated failure point (success rate < 95%): "
            f"{robustness['failure_point_users']} users"
        )
        lines.append("")

        lines.append("6) Edge Case Testing")
        for case in edge_cases:
            lines.append(
                "- {name}: status={status}, success={success}, latency={lat:.2f} ms, error={error}".format(
                    name=case["name"],
                    status=case["status_code"],
                    success=case["success"],
                    lat=case["latency_ms"],
                    error=case["error"],
                )
            )
        lines.append("")

        lines.append("7) Reproducibility Notes")
        lines.append("- Fixed request payload used for baseline tests.")
        lines.append("- Concurrency levels evaluated in deterministic user steps.")
        lines.append("- All outputs generated by a single script run and saved to this file.")

        return "\n".join(lines)

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = int(round((percentile / 100) * (len(ordered) - 1)))
        return ordered[idx]
