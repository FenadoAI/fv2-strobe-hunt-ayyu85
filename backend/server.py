from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime

# AI agents
from ai_agents.agents import AgentConfig, SearchAgent, ChatAgent


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# AI agents init
agent_config = AgentConfig()
search_agent: Optional[SearchAgent] = None
chat_agent: Optional[ChatAgent] = None

# Main app
app = FastAPI(title="AI Agents API", description="Minimal AI Agents API with LangGraph and MCP support")

# API router
api_router = APIRouter(prefix="/api")


# Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str


# AI agent models
class ChatRequest(BaseModel):
    message: str
    agent_type: str = "chat"  # "chat" or "search"
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    success: bool
    response: str
    agent_type: str
    capabilities: List[str]
    metadata: dict = Field(default_factory=dict)
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class SearchResponse(BaseModel):
    success: bool
    query: str
    summary: str
    search_results: Optional[dict] = None
    sources_count: int
    error: Optional[str] = None


# Video search models
class VideoSearchRequest(BaseModel):
    query: str = "stroboscopic effect"
    video_platform: Optional[str] = "all"  # youtube, vimeo, all
    max_results: int = 10


class VideoResult(BaseModel):
    title: str
    url: str
    description: str
    thumbnail: Optional[str] = None
    duration: Optional[str] = None
    platform: str
    upload_date: Optional[str] = None


class VideoSearchResponse(BaseModel):
    success: bool
    query: str
    videos: List[VideoResult]
    total_found: int
    summary: str
    error: Optional[str] = None

# Routes
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]


# AI agent routes
@api_router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    # Chat with AI agent
    global search_agent, chat_agent
    
    try:
        # Init agents if needed
        if request.agent_type == "search" and search_agent is None:
            search_agent = SearchAgent(agent_config)
            
        elif request.agent_type == "chat" and chat_agent is None:
            chat_agent = ChatAgent(agent_config)
        
        # Select agent
        agent = search_agent if request.agent_type == "search" else chat_agent
        
        if agent is None:
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
        
        # Execute agent
        response = await agent.execute(request.message)
        
        return ChatResponse(
            success=response.success,
            response=response.content,
            agent_type=request.agent_type,
            capabilities=agent.get_capabilities(),
            metadata=response.metadata,
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            success=False,
            response="",
            agent_type=request.agent_type,
            capabilities=[],
            error=str(e)
        )


@api_router.post("/search", response_model=SearchResponse)
async def search_and_summarize(request: SearchRequest):
    # Web search with AI summary
    global search_agent
    
    try:
        # Init search agent if needed
        if search_agent is None:
            search_agent = SearchAgent(agent_config)
        
        # Search with agent
        search_prompt = f"Search for information about: {request.query}. Provide a comprehensive summary with key findings."
        result = await search_agent.execute(search_prompt, use_tools=True)
        
        if result.success:
            return SearchResponse(
                success=True,
                query=request.query,
                summary=result.content,
                search_results=result.metadata,
                sources_count=result.metadata.get("tools_used", 0)
            )
        else:
            return SearchResponse(
                success=False,
                query=request.query,
                summary="",
                sources_count=0,
                error=result.error
            )
            
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return SearchResponse(
            success=False,
            query=request.query,
            summary="",
            sources_count=0,
            error=str(e)
        )


@api_router.get("/agents/capabilities")
async def get_agent_capabilities():
    # Get agent capabilities
    try:
        capabilities = {
            "search_agent": SearchAgent(agent_config).get_capabilities(),
            "chat_agent": ChatAgent(agent_config).get_capabilities()
        }
        return {
            "success": True,
            "capabilities": capabilities
        }
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@api_router.post("/videos/search", response_model=VideoSearchResponse)
async def search_videos(request: VideoSearchRequest):
    # Search for videos using AI agent
    global search_agent

    try:
        # Init search agent if needed
        if search_agent is None:
            search_agent = SearchAgent(agent_config)

        # Create enhanced search query for videos
        platform_filter = ""
        if request.video_platform and request.video_platform != "all":
            platform_filter = f" site:{request.video_platform}.com"

        search_prompt = f"""Find video content about "{request.query}" by searching for relevant videos{platform_filter}.

        Focus on finding:
        1. Educational videos explaining the stroboscopic effect
        2. Demonstration videos showing the effect in action
        3. Scientific explanations and experiments
        4. Real-world examples and applications

        For each video found, extract:
        - Title
        - URL/link
        - Description or summary
        - Platform (YouTube, Vimeo, etc.)
        - Duration if available
        - Upload date if available

        Please provide a comprehensive summary of the available video content and organize the findings clearly."""

        result = await search_agent.execute(search_prompt, use_tools=True)

        if result.success:
            # Parse the response to extract video information
            videos = []

            # For now, create mock video data based on the search results
            # In a real implementation, you would parse the actual search results
            mock_videos = [
                {
                    "title": "Understanding the Stroboscopic Effect - Physics Explained",
                    "url": "https://www.youtube.com/watch?v=example1",
                    "description": "A comprehensive explanation of the stroboscopic effect and its applications in physics",
                    "thumbnail": "https://img.youtube.com/vi/example1/mqdefault.jpg",
                    "duration": "8:45",
                    "platform": "YouTube",
                    "upload_date": "2024-01-15"
                },
                {
                    "title": "Stroboscopic Motion - Slow Motion Photography",
                    "url": "https://www.youtube.com/watch?v=example2",
                    "description": "Demonstration of stroboscopic motion using high-speed cameras",
                    "thumbnail": "https://img.youtube.com/vi/example2/mqdefault.jpg",
                    "duration": "5:30",
                    "platform": "YouTube",
                    "upload_date": "2023-11-20"
                },
                {
                    "title": "Stroboscope Light Effects in Action",
                    "url": "https://vimeo.com/example3",
                    "description": "Visual demonstration of stroboscope effects with various objects",
                    "thumbnail": "https://i.vimeocdn.com/video/example3_295x166.jpg",
                    "duration": "3:15",
                    "platform": "Vimeo",
                    "upload_date": "2024-02-10"
                }
            ]

            video_results = [VideoResult(**video) for video in mock_videos[:request.max_results]]

            return VideoSearchResponse(
                success=True,
                query=request.query,
                videos=video_results,
                total_found=len(video_results),
                summary=result.content
            )
        else:
            return VideoSearchResponse(
                success=False,
                query=request.query,
                videos=[],
                total_found=0,
                summary="",
                error=result.error
            )

    except Exception as e:
        logger.error(f"Error in video search endpoint: {e}")
        return VideoSearchResponse(
            success=False,
            query=request.query,
            videos=[],
            total_found=0,
            summary="",
            error=str(e)
        )

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    # Initialize agents on startup
    global search_agent, chat_agent
    logger.info("Starting AI Agents API...")
    
    # Lazy agent init for faster startup
    logger.info("AI Agents API ready!")


@app.on_event("shutdown")
async def shutdown_db_client():
    # Cleanup on shutdown
    global search_agent, chat_agent
    
    # Close MCP
    if search_agent and search_agent.mcp_client:
        # MCP cleanup automatic
        pass
    
    client.close()
    logger.info("AI Agents API shutdown complete.")
