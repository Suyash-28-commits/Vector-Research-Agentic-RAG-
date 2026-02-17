from fastapi import FastAPI,Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from src.research_agent_crew.crew import ResearchAgentCrew

app = FastAPI()

@app.get("/")
def home_page():
    try:
        return JSONResponse(
            status_code=200,
            content = {
                "welcome_message" : "Welcome to research agent crew",
                "message" : "Ask a query and I will let you query a research paper"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occured {str(e)}"
        )

@app.get("/research")
async def research(query: str = Query(...,description="User's query")):
    try:
        crew_instance = ResearchAgentCrew()
        crew = crew_instance.crew()
        response = await run_in_threadpool(
            crew.kickoff,
            {'query':query}
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail = f"An error occured: {str(e)}"
        )
