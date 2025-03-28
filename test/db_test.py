from dotenv import load_dotenv
import os

from supabase.client import create_client

# Load .env from project root (adjust path if needed)
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

url = os.getenv("SUPABASE_URL")
print(url)
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Test connection
data = supabase.table("interactions").select("*").limit(1).execute()
print(data)