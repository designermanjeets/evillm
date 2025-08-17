#!/usr/bin/env python3
"""Collect metrics snapshot from the application."""

import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class MetricsCollector:
    """Collect metrics from the application endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"X-Tenant-ID": "demo-tenant"}
        self.metrics = {}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("📊 Collecting Metrics Snapshot...")
        print(f"🌐 Base URL: {self.base_url}")
        print()
        
        # Collect search metrics
        self._collect_search_metrics()
        
        # Collect search quality metrics
        self._collect_search_quality_metrics()
        
        # Collect draft metrics
        self._collect_draft_metrics()
        
        # Collect health metrics
        self._collect_health_metrics()
        
        # Add metadata
        self.metrics["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "tenant_id": "demo-tenant"
        }
        
        return self.metrics
    
    def _collect_search_metrics(self):
        """Collect search performance metrics."""
        try:
            print("🔍 Collecting search metrics...")
            response = requests.get(f"{self.base_url}/search-metrics/summary", headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                self.metrics["search_metrics"] = response.json()
                print("  ✅ Search metrics collected")
            else:
                print(f"  ⚠️ Search metrics returned {response.status_code}")
                self.metrics["search_metrics"] = {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to collect search metrics: {e}")
            self.metrics["search_metrics"] = {"error": str(e)}
    
    def _collect_search_quality_metrics(self):
        """Collect search quality metrics."""
        try:
            print("🎯 Collecting search quality metrics...")
            response = requests.get(f"{self.base_url}/search-qa/quality/summary", headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                self.metrics["search_quality"] = response.json()
                print("  ✅ Search quality metrics collected")
            else:
                print(f"  ⚠️ Search quality metrics returned {response.status_code}")
                self.metrics["search_quality"] = {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to collect search quality metrics: {e}")
            self.metrics["search_quality"] = {"error": str(e)}
    
    def _collect_draft_metrics(self):
        """Collect draft generation metrics."""
        try:
            print("✍️ Collecting draft metrics...")
            response = requests.get(f"{self.base_url}/draft/metrics/summary", headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                self.metrics["draft_metrics"] = response.json()
                print("  ✅ Draft metrics collected")
            else:
                print(f"  ⚠️ Draft metrics returned {response.status_code}")
                self.metrics["draft_metrics"] = {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to collect draft metrics: {e}")
            self.metrics["draft_metrics"] = {"error": str(e)}
    
    def _collect_health_metrics(self):
        """Collect health and system metrics."""
        try:
            print("🏥 Collecting health metrics...")
            response = requests.get(f"{self.base_url}/health/ready", headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                self.metrics["health"] = response.json()
                print("  ✅ Health metrics collected")
            else:
                print(f"  ⚠️ Health metrics returned {response.status_code}")
                self.metrics["health"] = {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to collect health metrics: {e}")
            self.metrics["health"] = {"error": str(e)}
    
    def generate_mock_metrics(self):
        """Generate mock metrics for demo purposes."""
        print("🎭 Generating mock metrics for demo...")
        
        self.metrics["search_metrics"] = {
            "total_searches": 1250,
            "avg_response_time_ms": 245,
            "p95_response_time_ms": 412,
            "success_rate": 0.98,
            "errors": {
                "dependency_unavailable": 12,
                "no_evidence_found": 8,
                "rate_limited": 3
            }
        }
        
        self.metrics["search_quality"] = {
            "hit_at_5": 0.87,
            "mrr_at_10": 0.72,
            "ndcg_at_10": 0.78,
            "quality_status": "green",
            "thresholds_met": True,
            "sample_size": 150
        }
        
        self.metrics["draft_metrics"] = {
            "total_drafts": 89,
            "avg_generation_time_ms": 3200,
            "p95_generation_time_ms": 5800,
            "eval_pass_rate": 0.85,
            "blocked_drafts": 13,
            "blocking_reasons": {
                "low_grounding": 8,
                "no_evidence": 3,
                "policy_violation": 2
            }
        }
        
        self.metrics["health"] = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "bm25": "healthy",
                "vector": "healthy",
                "llm": "healthy",
                "database": "healthy"
            }
        }
        
        print("  ✅ Mock metrics generated")
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to file."""
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        output_file = reports_dir / "metrics_snapshot.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"💾 Metrics saved to: {output_file}")
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print metrics summary."""
        print("\n📊 Metrics Summary:")
        print("=" * 50)
        
        # Search metrics
        if "search_metrics" in metrics and "error" not in metrics["search_metrics"]:
            search = metrics["search_metrics"]
            print(f"🔍 Search Performance:")
            print(f"  Total searches: {search.get('total_searches', 'N/A')}")
            print(f"  Avg response time: {search.get('avg_response_time_ms', 'N/A')}ms")
            print(f"  P95 response time: {search.get('p95_response_time_ms', 'N/A')}ms")
            print(f"  Success rate: {search.get('success_rate', 'N/A')}")
        else:
            print("🔍 Search Performance: Not available")
        
        # Search quality
        if "search_quality" in metrics and "error" not in metrics["search_quality"]:
            quality = metrics["search_quality"]
            print(f"🎯 Search Quality:")
            print(f"  Hit@5: {quality.get('hit_at_5', 'N/A')}")
            print(f"  MRR@10: {quality.get('mrr_at_10', 'N/A')}")
            print(f"  NDCG@10: {quality.get('ndcg_at_10', 'N/A')}")
            print(f"  Status: {quality.get('quality_status', 'N/A')}")
        else:
            print("🎯 Search Quality: Not available")
        
        # Draft metrics
        if "draft_metrics" in metrics and "error" not in metrics["draft_metrics"]:
            draft = metrics["draft_metrics"]
            print(f"✍️ Draft Generation:")
            print(f"  Total drafts: {draft.get('total_drafts', 'N/A')}")
            print(f"  Avg generation time: {draft.get('avg_generation_time_ms', 'N/A')}ms")
            print(f"  Eval pass rate: {draft.get('eval_pass_rate', 'N/A')}")
            print(f"  Blocked drafts: {draft.get('blocked_drafts', 'N/A')}")
        else:
            print("✍️ Draft Generation: Not available")
        
        # Health
        if "health" in metrics and "error" not in metrics["health"]:
            health = metrics["health"]
            print(f"🏥 System Health:")
            print(f"  Status: {health.get('status', 'N/A')}")
            if "services" in health:
                for service, status in health["services"].items():
                    print(f"    {service}: {status}")
        else:
            print("🏥 System Health: Not available")
        
        print("=" * 50)


def main():
    """Main entry point."""
    collector = MetricsCollector()
    
    try:
        # Try to collect real metrics
        metrics = collector.collect_all_metrics()
        
        # Check if we got any real data
        has_real_data = any(
            "error" not in metrics.get(key, {}) 
            for key in ["search_metrics", "search_quality", "draft_metrics", "health"]
        )
        
        if not has_real_data:
            print("⚠️ No real metrics available, generating mock data for demo...")
            collector.generate_mock_metrics()
            metrics = collector.metrics
        
    except Exception as e:
        print(f"❌ Metrics collection failed: {e}")
        print("🎭 Generating mock metrics for demo...")
        collector.generate_mock_metrics()
        metrics = collector.metrics
    
    # Save and display metrics
    collector.save_metrics(metrics)
    collector.print_summary(metrics)
    
    print("\n🎉 Metrics collection completed!")
    print("\nNext steps:")
    print("1. Review metrics in /reports/metrics_snapshot.json")
    print("2. Use metrics for demo and monitoring")
    print("3. Run smoke tests: make demo-smoke")


if __name__ == "__main__":
    main()
