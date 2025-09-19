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
import asyncio

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
    # Search for videos using AI agent with real video search
    global search_agent

    try:
        # Init search agent if needed
        if search_agent is None:
            search_agent = SearchAgent(agent_config)

        # Create enhanced search query for videos
        platform_filter = ""
        if request.video_platform and request.video_platform != "all":
            platform_filter = f" site:{request.video_platform}.com"

        search_prompt = f"""Quick search for "{request.query}" videos. Find REAL YouTube URLs only.
        Return actual video URLs in format: https://www.youtube.com/watch?v=VIDEO_ID
        Be concise - I need working links quickly."""

        # Add timeout to search
        try:
            result = await asyncio.wait_for(
                search_agent.execute(search_prompt, use_tools=True),
                timeout=45.0  # 45 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Search timed out for query: {request.query}")
            return VideoSearchResponse(
                success=False,
                query=request.query,
                videos=[],
                total_found=0,
                summary="Search timed out - please try a different query",
                error="Search timeout"
            )

        if result.success:
            # Parse the AI response to extract video information
            videos = []

            # Try to extract structured video data from the search results
            # This is a basic implementation - in production you'd use more sophisticated parsing
            response_text = result.content.lower()

            # Look for video URLs in the response
            import re

            # Find YouTube URLs
            youtube_urls = re.findall(r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', result.content)

            # Find Vimeo URLs
            vimeo_urls = re.findall(r'https?://(?:www\.)?vimeo\.com/(\d+)', result.content)

            # Create structured video data from found URLs
            video_count = 0

            # Extract titles from the response text
            title_patterns = []
            lines = result.content.split('\n')
            for line in lines:
                if 'youtube.com/watch?v=' in line and any(word in line.lower() for word in ['title', '**', '*', '-']):
                    title_patterns.append(line.strip('* -:').strip())

            for i, video_id in enumerate(youtube_urls[:request.max_results]):
                if video_count >= request.max_results:
                    break

                # Try to extract title from the search results
                title = f"Educational Video: {request.query.title()}"
                if i < len(title_patterns):
                    # Clean up the title from the AI response
                    extracted_title = title_patterns[i].split('by')[0].split(':')[0].strip('*').strip()
                    if extracted_title and len(extracted_title) > 10:
                        title = extracted_title

                videos.append(VideoResult(
                    title=title,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    description=f"Real educational video about {request.query} discovered through AI search",
                    thumbnail=f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                    platform="YouTube",
                    duration=None,
                    upload_date=None
                ))
                video_count += 1

            for video_id in vimeo_urls[:max(0, request.max_results - video_count)]:
                if video_count >= request.max_results:
                    break

                videos.append(VideoResult(
                    title=f"Video about {request.query} - Real Content",
                    url=f"https://vimeo.com/{video_id}",
                    description=f"Educational content about {request.query} found through AI search",
                    thumbnail="https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=600&h=400&fit=crop",
                    platform="Vimeo",
                    duration=None,
                    upload_date=None
                ))
                video_count += 1

            # If no URLs found, create a structured response based on search content
            if not videos:
                # Fallback: create entries based on search results but with real search-based info
                videos = [
                    VideoResult(
                        title=f"Search Results: {request.query}",
                        url=f"https://www.youtube.com/results?search_query={request.query.replace(' ', '+')}",
                        description=f"Search results for {request.query} based on AI analysis",
                        thumbnail="https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=600&h=400&fit=crop",
                        platform="YouTube Search",
                        duration=None,
                        upload_date=None
                    )
                ]

            return VideoSearchResponse(
                success=True,
                query=request.query,
                videos=videos,
                total_found=len(videos),
                summary=result.content,
                error=None
            )
        else:
            return VideoSearchResponse(
                success=False,
                query=request.query,
                videos=[],
                total_found=0,
                summary="Search failed",
                error=result.error
            )

    except Exception as e:
        logger.error(f"Error in video search endpoint: {e}")
        return VideoSearchResponse(
            success=False,
            query=request.query,
            videos=[],
            total_found=0,
            summary="Search error occurred",
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
