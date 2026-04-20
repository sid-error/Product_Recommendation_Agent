import os
import sys
import json
import logging

# CRITICAL: Suppress ALL FastMCP/rich/mcp logging and banner output.
# FastMCP prints ASCII banners and "Starting MCP server" messages to stdout,
# which corrupts the JSON-RPC stdio protocol used by ADK's McpToolset.
logging.disable(logging.CRITICAL)
os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"
os.environ["NO_COLOR"] = "1"

import pandas as pd
from typing import List, Optional

try:
    from fastmcp import FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("Sales Catalog Server")

# Load data into a DataFrame at startup
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "product_info.csv"))
try:
    df = pd.read_csv(csv_path)
    if 'productId' in df.columns:
        df['productId'] = df['productId'].astype(str)
except Exception as e:
    df = pd.DataFrame()
    print(f"Warning: Could not load CSV data: {e}", file=sys.stderr)

@mcp.tool()
def get_taxonomy() -> str:
    """Returns a unique list of dictionaries containing available category and subCategory pairs."""
    if df.empty:
        return json.dumps({"error": "No data available."})
    
    if 'category' not in df.columns or 'subCategory' not in df.columns:
         return json.dumps({"error": "Schema missing category or subCategory columns."})

    taxonomy_df = df[['category', 'subCategory']].dropna().drop_duplicates()
    taxonomy_list = taxonomy_df.to_dict(orient='records')
    return json.dumps(taxonomy_list)

@mcp.tool()
def search_products(merchantDivision: Optional[str] = None, category: Optional[str] = None, subCategory: Optional[str] = None, articleType: Optional[str] = None, max_price: Optional[float] = None) -> str:
    """Filters products by classifiers or price. Prioritizes items where offerPrice <= max_price. Returns max 5 best matches with full details including imagery."""
    if df.empty:
        return json.dumps({"error": "No data available."})
    
    results = df.copy()
    
    if merchantDivision:
        results = results[results['merchantDivision'].str.contains(merchantDivision, case=False, na=False)]
    if category:
        results = results[results['category'].str.contains(category, case=False, na=False)]
    if subCategory:
        results = results[results['subCategory'].str.contains(subCategory, case=False, na=False)]
    if articleType:
        results = results[results['articleType'].str.contains(articleType, case=False, na=False)]
        
    if max_price is not None:
        mask_affordable = results['offerPrice'] <= max_price
        results['is_affordable'] = mask_affordable
        results = results.sort_values(by=['is_affordable', 'offerPrice'], ascending=[False, True])
    else:
        results = results.sort_values(by=['offerPrice'], ascending=True)

    top_5 = results.head(5)
    
    cols_to_drop = [col for col in ['is_affordable'] if col in top_5.columns]
    if cols_to_drop:
        top_5 = top_5.drop(columns=cols_to_drop)
        
    return top_5.to_json(orient='records')

@mcp.tool()
def get_product_details(product_ids: List[str]) -> str:
    """Takes a list of product IDs and returns the full details, crucially including imageUrl, price, and offerPrice."""
    if df.empty:
        return json.dumps({"error": "No data available."})
    
    results = df[df['productId'].isin(product_ids)]
    return results.to_json(orient='records')

if __name__ == "__main__":
    import asyncio
    
    # Bypass FastMCP's run_stdio_async() which prints banners.
    # Use the raw mcp library's stdio transport directly.
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options()
            )
    
    asyncio.run(main())
