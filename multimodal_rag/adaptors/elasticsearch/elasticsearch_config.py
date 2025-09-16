"""Configuration for Elasticsearch adaptor."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class ElasticsearchConfig(BaseModel):
    """Configuration for Elasticsearch connection and indices."""
    
    # Connection settings
    hosts: list[str] = Field(default=["http://localhost:9200"])
    username: Optional[str] = None
    password: Optional[str] = None
    verify_certs: bool = True
    use_ssl: bool = False
    ca_certs: Optional[str] = None
    
    # Index settings
    index_name: str = Field(default="multimodal_index", description="Unified index for documents and chunks")
    vector_dimensions: int = Field(default=1536, description="Dimension of embedding vectors")
    
    # Index configurations
    shards: int = Field(default=1)
    replicas: int = Field(default=0)
    
    # Search settings
    default_search_size: int = Field(default=10, ge=1, le=100)
    max_search_size: int = Field(default=100, ge=1, le=1000)
    
    def get_es_config(self) -> Dict[str, Any]:
        """Get Elasticsearch client configuration."""
        config = {
            "hosts": self.hosts,
            "verify_certs": self.verify_certs,
        }
        
        if self.username and self.password:
            config["http_auth"] = (self.username, self.password)
        
        if self.use_ssl:
            config["use_ssl"] = True
            if self.ca_certs:
                config["ca_certs"] = self.ca_certs
        
        return config
    
    def get_index_settings(self) -> Dict[str, Any]:
        """Get settings for the unified index."""
        return {
            "settings": {
                "number_of_shards": self.shards,
                "number_of_replicas": self.replicas,
                "analysis": {
                    "analyzer": {
                        "multimodal_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            }
        }
