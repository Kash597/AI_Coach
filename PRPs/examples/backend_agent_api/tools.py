from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_globals, safe_builtins, guarded_unpack_sequence
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, BinaryContent
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from httpx import AsyncClient
from supabase import Client
import base64
import json
import sys
import os
import re

embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE') or 'text-embedding-3-small'

async def get_embedding(text: str, embedding_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def retrieve_relevant_documents_tool(supabase: Client, embedding_client: AsyncOpenAI, user_query: str) -> str:
    """
    Function to retrieve relevant document chunks with RAG.
    This is called by the retrieve_relevant_documents tool for the agent.
    
    Returns:
        List[str]: List of relevant document chunks with metadata
    """    
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, embedding_client)
        
        # Query Supabase for relevant documents
        result = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': 4
            }
        ).execute()
        
        if not result.data:
            return "No relevant documents found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# Document ID: {doc['metadata'].get('file_id', 'unknown')}      
# Document Tilte: {doc['metadata'].get('file_title', 'unknown')}
# Document URL: {doc['metadata'].get('file_url', 'unknown')}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return f"Error retrieving documents: {str(e)}" 

async def list_documents_tool(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available documents.
    This is called by the list_documents tool for the agent.
    
    Returns:
        List[str]: List of documents including their metadata (URL/path, schema if applicable, etc.)
    """
    try:
        # Query Supabase for unique documents
        result = supabase.from_('document_metadata') \
            .select('id, title, schema, url') \
            .execute()
            
        return str(result.data)
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return str([])

async def get_document_content_tool(supabase: Client, document_id: str) -> str:
    """
    Retrieve the full content of a specific document by combining all its chunks.
    This is called by the get_document_content tool for the agent.
        
    Returns:
        str: The complete content of the document with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks for this document
        result = supabase.from_('documents') \
            .select('id, content, metadata') \
            .eq('metadata->>file_id', document_id) \
            .order('id') \
            .execute()
        
        if not result.data:
            return f"No content found for document: {document_id}"
            
        # Format the document with its title and all chunks
        document_title = result.data[0]['metadata']['file_title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {document_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together but limit the characters in case the document is massive
        return "\n\n".join(formatted_content)[:20000]
        
    except Exception as e:
        print(f"Error retrieving document content: {e}")
        return f"Error retrieving document content: {str(e)}"     
