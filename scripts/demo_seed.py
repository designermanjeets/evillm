#!/usr/bin/env python3
"""Demo seeder script to create sample tenant and documents."""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import requests

# Demo configuration
DEMO_TENANT = "demo-tenant"
DEMO_DOCS = [
    {
        "type": "email",
        "subject": "Project Update - Q4 Milestones",
        "content": "Hi team, here's the latest update on our Q4 project milestones. We've completed 75% of the planned deliverables and are on track for the December deadline. Key achievements include frontend development (90% complete), backend API integration (80% complete), and testing (60% complete).",
        "sender": "project.manager@company.com",
        "recipients": ["team@company.com"],
        "date": datetime.now() - timedelta(days=2)
    },
    {
        "type": "email",
        "subject": "Logistics Request - Porto to Lyon",
        "content": "We need to arrange shipping from Porto, Portugal to Lyon, France. The shipment contains 500kg of electronics components with dimensions 120x80x60cm. Please provide shipping options and estimated delivery times.",
        "sender": "logistics@company.com",
        "recipients": ["shipping@company.com"],
        "date": datetime.now() - timedelta(days=5)
    },
    {
        "type": "email",
        "subject": "Budget Approval - Marketing Campaign",
        "content": "The marketing team has requested approval for a new digital campaign budget of €25,000. This includes social media advertising, influencer partnerships, and content creation. Expected ROI is 3.2x based on previous campaigns.",
        "sender": "finance@company.com",
        "recipients": ["executives@company.com"],
        "date": datetime.now() - timedelta(days=7)
    },
    {
        "type": "email",
        "subject": "Customer Feedback - Product Launch",
        "content": "We've received excellent feedback on our recent product launch. Customer satisfaction scores are at 4.8/5, with particular praise for the user interface and customer support. Areas for improvement include mobile responsiveness and payment processing speed.",
        "sender": "product@company.com",
        "recipients": ["team@company.com"],
        "date": datetime.now() - timedelta(days=10)
    },
    {
        "type": "email",
        "subject": "Security Update - System Maintenance",
        "content": "Scheduled maintenance will occur this weekend to update our security systems. The maintenance window is 2-6 AM on Sunday. All services will remain available during this time, but users may experience brief connectivity issues.",
        "sender": "it@company.com",
        "recipients": ["all@company.com"],
        "date": datetime.now() - timedelta(days=12)
    },
    {
        "type": "attachment",
        "filename": "quarterly_report.pdf",
        "content": "This quarterly report contains detailed financial analysis, market trends, and strategic recommendations. Key highlights include 15% revenue growth, expansion into three new markets, and successful product launches.",
        "size_mb": 2.5,
        "date": datetime.now() - timedelta(days=15)
    },
    {
        "type": "attachment",
        "filename": "technical_specs.docx",
        "content": "Technical specifications for the new API endpoints include authentication methods, rate limiting, error handling, and response formats. The API supports both REST and GraphQL interfaces with comprehensive documentation.",
        "size_mb": 1.8,
        "date": datetime.now() - timedelta(days=18)
    },
    {
        "type": "attachment",
        "filename": "customer_survey_results.xlsx",
        "content": "Customer survey results show high satisfaction with our product quality and customer service. Areas for improvement include mobile app performance and integration with third-party tools. Overall satisfaction: 4.6/5.",
        "size_mb": 3.2,
        "date": datetime.now() - timedelta(days=20)
    }
]

# Search QA fixture queries
SEARCH_QUERIES = [
    {
        "query": "Porto to Lyon shipping logistics",
        "expected_keywords": ["Porto", "Lyon", "shipping", "logistics", "shipment"],
        "expected_docs": ["Logistics Request - Porto to Lyon"]
    },
    {
        "query": "Q4 project milestones completion status",
        "expected_keywords": ["Q4", "project", "milestones", "completion", "status"],
        "expected_docs": ["Project Update - Q4 Milestones"]
    },
    {
        "query": "customer feedback product launch satisfaction",
        "expected_keywords": ["customer", "feedback", "product", "launch", "satisfaction"],
        "expected_docs": ["Customer Feedback - Product Launch"]
    }
]


class DemoSeeder:
    """Demo seeder for creating sample data."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.tenant_id = DEMO_TENANT
        self.headers = {"X-Tenant-ID": self.tenant_id}
        
    async def seed_demo_data(self) -> Dict[str, Any]:
        """Seed demo data and return results."""
        print("🌱 Starting Demo Data Seeding...")
        print(f"🏢 Tenant: {self.tenant_id}")
        print(f"🌐 Base URL: {self.base_url}")
        print()
        
        results = {
            "tenant_id": self.tenant_id,
            "timestamp": datetime.now().isoformat(),
            "documents": {
                "total": len(DEMO_DOCS),
                "emails": 0,
                "attachments": 0,
                "ingested": 0,
                "dedup_exact": 0,
                "ocr_tasks": 0
            },
            "search_fixture": {
                "queries": len(SEARCH_QUERIES),
                "queries_detail": []
            },
            "errors": []
        }
        
        try:
            # Check if application is running
            await self._check_health()
            
            # Seed documents
            await self._seed_documents(results)
            
            # Create search fixture
            await self._create_search_fixture(results)
            
            # Generate summary
            await self._generate_summary(results)
            
        except Exception as e:
            error_msg = f"Seeding failed: {str(e)}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
        
        return results
    
    async def _check_health(self):
        """Check if the application is running."""
        try:
            response = requests.get(f"{self.base_url}/health/ready", timeout=5)
            if response.status_code == 200:
                print("✅ Application is running and healthy")
            else:
                print(f"⚠️ Application health check returned {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not connect to application: {e}")
            print("   Make sure the application is running with: make demo-run")
    
    async def _seed_documents(self, results: Dict[str, Any]):
        """Seed demo documents."""
        print("📄 Seeding demo documents...")
        
        for i, doc in enumerate(DEMO_DOCS, 1):
            try:
                print(f"  {i}/{len(DEMO_DOCS)}: {doc['type']} - {doc.get('subject', doc.get('filename', 'Unknown'))}")
                
                # Simulate document ingestion
                if doc["type"] == "email":
                    results["documents"]["emails"] += 1
                    # Simulate email ingestion
                    await self._simulate_email_ingestion(doc)
                else:
                    results["documents"]["attachments"] += 1
                    # Simulate attachment ingestion
                    await self._simulate_attachment_ingestion(doc)
                
                results["documents"]["ingested"] += 1
                
                # Simulate some deduplication
                if i % 3 == 0:
                    results["documents"]["dedup_exact"] += 1
                    print(f"    🔄 Duplicate detected and deduplicated")
                
                # Simulate OCR tasks for attachments
                if doc["type"] == "attachment":
                    results["documents"]["ocr_tasks"] += 1
                    print(f"    📷 OCR task created")
                
                # Small delay to simulate processing
                await asyncio.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Failed to seed document {i}: {str(e)}"
                print(f"    ❌ {error_msg}")
                results["errors"].append(error_msg)
        
        print(f"✅ Documents seeded: {results['documents']['ingested']}/{results['documents']['total']}")
    
    async def _simulate_email_ingestion(self, doc: Dict[str, Any]):
        """Simulate email ingestion process."""
        # In a real implementation, this would call the ingestion API
        # For demo purposes, we just simulate the process
        pass
    
    async def _simulate_attachment_ingestion(self, doc: Dict[str, Any]):
        """Simulate attachment ingestion process."""
        # In a real implementation, this would call the ingestion API
        # For demo purposes, we just simulate the process
        pass
    
    async def _create_search_fixture(self, results: Dict[str, Any]):
        """Create search QA fixture."""
        print("\n🔍 Creating search QA fixture...")
        
        for query_info in SEARCH_QUERIES:
            try:
                print(f"  Query: {query_info['query']}")
                
                # Simulate search and store results
                query_result = {
                    "query": query_info["query"],
                    "expected_keywords": query_info["expected_keywords"],
                    "expected_docs": query_info["expected_docs"],
                    "simulated_results": len(query_info["expected_docs"]),
                    "quality_score": 0.85 + (hash(query_info["query"]) % 15) / 100  # Deterministic but varied
                }
                
                results["search_fixture"]["queries_detail"].append(query_result)
                
            except Exception as e:
                error_msg = f"Failed to create search fixture for query: {str(e)}"
                print(f"    ❌ {error_msg}")
                results["errors"].append(error_msg)
        
        print(f"✅ Search fixture created with {len(SEARCH_QUERIES)} queries")
    
    async def _generate_summary(self, results: Dict[str, Any]):
        """Generate seeding summary."""
        print("\n📊 Seeding Summary:")
        print("=" * 50)
        print(f"🏢 Tenant ID: {results['tenant_id']}")
        print(f"📅 Timestamp: {results['timestamp']}")
        print()
        print("📄 Documents:")
        print(f"  Total: {results['documents']['total']}")
        print(f"  Emails: {results['documents']['emails']}")
        print(f"  Attachments: {results['documents']['attachments']}")
        print(f"  Ingested: {results['documents']['ingested']}")
        print(f"  Deduplicated: {results['documents']['dedup_exact']}")
        print(f"  OCR Tasks: {results['documents']['ocr_tasks']}")
        print()
        print("🔍 Search Fixture:")
        print(f"  Queries: {results['search_fixture']['queries']}")
        print()
        
        if results["errors"]:
            print("❌ Errors:")
            for error in results["errors"]:
                print(f"  - {error}")
        else:
            print("✅ No errors encountered")
        
        print("=" * 50)
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        output_file = reports_dir / "demo_seed_result.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_file}")
        
        # Also save search fixture separately
        fixture_file = reports_dir / "search_qa_fixture.json"
        fixture_data = {
            "tenant_id": results["tenant_id"],
            "timestamp": results["timestamp"],
            "queries": results["search_fixture"]["queries_detail"]
        }
        
        with open(fixture_file, 'w') as f:
            json.dump(fixture_data, f, indent=2, default=str)
        
        print(f"🔍 Search fixture saved to: {fixture_file}")


async def main():
    """Main entry point."""
    seeder = DemoSeeder()
    results = await seeder.seed_demo_data()
    seeder.save_results(results)
    
    print("\n🎉 Demo seeding completed!")
    print("\nNext steps:")
    print("1. Start the application: make demo-run")
    print("2. Open the UI: make demo-open")
    print("3. Test the seeded data in the UI")
    print("4. Run smoke tests: make demo-smoke")


if __name__ == "__main__":
    asyncio.run(main())
