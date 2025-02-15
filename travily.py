from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1. Instantiating your TavilyClient
travily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Step 2. Executing the search request
response = travily_client.search("Who is Leo Messi?", max_results=10)

# Step 3. Printing the search results
for result in response["results"]:
    print(result["url"])