# example_usage.py
import asyncio
from main import RiskRAGSystem

async def main():
    # Initialize system using factory pattern
    system = await RiskRAGSystem.create()
    await system.start()

    try:
        # Example query
        response = await system.process_query(
            query="Tell me about the risks in this project",
            context={
                "query_type": "risk_analysis",  # This matches our metric labels
                "filters": {"type": "risk"},
                "top_k": 5
            }
        )
        print("Response:", response)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())