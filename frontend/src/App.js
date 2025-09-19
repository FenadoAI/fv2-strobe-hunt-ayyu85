import { useEffect, useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { Button } from "./components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Badge } from "./components/ui/badge";
import { Search, Play, Clock, Calendar, ExternalLink } from "lucide-react";

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API = `${API_BASE}/api`;

const Home = () => {
  const [searchQuery, setSearchQuery] = useState("stroboscopic effect");
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState("");
  const [totalFound, setTotalFound] = useState(0);

  const searchVideos = async (query = searchQuery) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/videos/search`, {
        query: query,
        video_platform: "all",
        max_results: 10
      });

      if (response.data.success) {
        setVideos(response.data.videos || []);
        setSummary(response.data.summary || "");
        setTotalFound(response.data.total_found || 0);
      } else {
        console.error("Search failed:", response.data.error);
        setVideos([]);
        setSummary("");
        setTotalFound(0);
      }
    } catch (error) {
      console.error("Error searching videos:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    searchVideos();
  }, []);

  const handleSearch = (e) => {
    e.preventDefault();
    searchVideos();
  };

  const VideoCard = ({ video }) => {
    const [imageError, setImageError] = useState(false);
    const [imageLoaded, setImageLoaded] = useState(false);

    const handleImageError = () => {
      setImageError(true);
    };

    const handleImageLoad = () => {
      setImageLoaded(true);
    };

    const getThumbnailUrl = () => {
      if (imageError || !video.thumbnail) {
        return "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=600&h=400&fit=crop";
      }
      return video.thumbnail;
    };

    return (
      <Card className="overflow-hidden hover:shadow-lg transition-all duration-300 group bg-white/5 backdrop-blur-sm border-white/10">
        <div className="relative">
          {!imageLoaded && (
            <div className="w-full h-48 bg-gray-800 animate-pulse flex items-center justify-center">
              <div className="text-gray-500">Loading...</div>
            </div>
          )}
          <img
            src={getThumbnailUrl()}
            alt={video.title}
            className={`w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300 ${
              !imageLoaded ? 'opacity-0' : 'opacity-100'
            }`}
            onError={handleImageError}
            onLoad={handleImageLoad}
          />
          <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors duration-300" />
          <Button
            size="icon"
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-white/20 backdrop-blur-sm hover:bg-white/30"
            onClick={() => window.open(video.url, '_blank')}
          >
            <Play className="h-6 w-6 text-white" />
          </Button>
        </div>
        <CardHeader>
          <div className="flex justify-between items-start gap-2">
            <CardTitle className="text-lg line-clamp-2 group-hover:text-purple-400 transition-colors text-white">
              {video.title}
            </CardTitle>
            <Badge variant="secondary" className="shrink-0 bg-purple-600/20 text-purple-300 border-purple-500/30">
              {video.platform}
            </Badge>
          </div>
          <CardDescription className="line-clamp-3 text-gray-300">
            {video.description}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 text-sm text-gray-400">
            {video.duration && (
              <div className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                {video.duration}
              </div>
            )}
            {video.upload_date && (
              <div className="flex items-center gap-1">
                <Calendar className="h-4 w-4" />
                {new Date(video.upload_date).toLocaleDateString()}
              </div>
            )}
          </div>
        </CardContent>
        <CardFooter>
          <Button
            variant="outline"
            className="w-full border-white/20 text-white hover:bg-white/10 hover:text-purple-300"
            onClick={() => window.open(video.url, '_blank')}
          >
            <ExternalLink className="h-4 w-4 mr-2" />
            Watch Video
          </Button>
        </CardFooter>
      </Card>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Hero Section */}
      <div
        className="relative h-96 flex items-center justify-center overflow-hidden"
        style={{
          backgroundImage: `linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url(https://storage.googleapis.com/fenado-ai-farm-public/generated/44832ab5-fb80-4c74-8298-c4b3930104d7.webp)`,
          backgroundSize: 'cover',
          backgroundPosition: 'center'
        }}
      >
        <div className="text-center z-10 max-w-4xl mx-auto px-6">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight">
            Stroboscopic
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
              Video Search
            </span>
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Discover fascinating videos about the stroboscopic effect - from scientific explanations to stunning visual demonstrations
          </p>

          {/* Search Form */}
          <form onSubmit={handleSearch} className="flex gap-4 max-w-md mx-auto">
            <Input
              type="text"
              placeholder="Search for videos..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 h-12 text-lg bg-white/10 backdrop-blur-sm border-white/20 text-white placeholder:text-gray-300"
            />
            <Button
              type="submit"
              size="lg"
              disabled={loading}
              className="h-12 px-6 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
            >
              {loading ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              ) : (
                <Search className="h-5 w-5" />
              )}
            </Button>
          </form>
        </div>
      </div>

      {/* Results Section */}
      <div className="container mx-auto px-6 py-12">
        {totalFound > 0 && (
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-white mb-4">
              Found {totalFound} Videos
            </h2>
            {summary && (
              <div className="max-w-4xl mx-auto">
                <Card className="bg-white/10 backdrop-blur-sm border-white/20">
                  <CardContent className="p-6">
                    <p className="text-gray-300 text-lg leading-relaxed">
                      {summary}
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        )}

        {/* Video Grid */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <Card key={i} className="animate-pulse bg-white/5 backdrop-blur-sm border-white/10">
                <div className="h-48 bg-gray-700 rounded-t-lg"></div>
                <CardHeader>
                  <div className="h-4 bg-gray-600 rounded w-3/4 mb-2"></div>
                  <div className="h-3 bg-gray-600 rounded w-1/2"></div>
                </CardHeader>
              </Card>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {videos.map((video, index) => (
              <VideoCard key={`${video.url}-${index}`} video={video} />
            ))}
          </div>
        )}

        {videos.length === 0 && !loading && (
          <div className="text-center py-12">
            <h3 className="text-2xl font-semibold text-white mb-4">
              No videos found
            </h3>
            <p className="text-gray-400">
              Try searching with different keywords
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-black/20 backdrop-blur-sm border-t border-white/10 py-8">
        <div className="container mx-auto px-6 text-center">
          <p className="text-gray-400">
            Powered by AI-driven search technology for discovering educational video content
          </p>
        </div>
      </footer>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />}>
            <Route index element={<Home />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
