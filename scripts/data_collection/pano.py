import requests
import platform
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
import sys
import progressbar
import math
from typing import Dict, Optional
from datetime import datetime

def fetch_and_process_json(url: str) -> Optional[Dict]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error retrieving data: {e}")
        return None

async def fetch_tile(session: aiohttp.ClientSession, url: str, retries: int = 2) -> Optional[Image.Image]:
    """Загружает тайл с повторными попытками"""
    for attempt in range(retries + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:

                if response.status == 429:
                    if attempt < retries:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    return None
                
                response.raise_for_status()
                content = await response.read()
                if len(content) == 0:
                    if attempt < retries:
                        await asyncio.sleep(0.3)
                        continue
                    return None
                
                img = Image.open(BytesIO(content))

                if img.size[0] == 0 or img.size[1] == 0:
                    if attempt < retries:
                        await asyncio.sleep(0.3)
                        continue
                    return None
                

                if img.mode == 'RGB':
                    pixels = list(img.getdata())
                    if len(pixels) > 0:
                        black_pixels = sum(1 for p in pixels if p == (0, 0, 0))

                        if black_pixels == len(pixels):
                            if attempt < retries:
                                await asyncio.sleep(0.3)
                                continue
                            return None
                
                return img
        except aiohttp.ClientResponseError as e:

            if e.status == 429:  
                if attempt < retries:
                    wait_time = 2.0 * (attempt + 1)  
                    await asyncio.sleep(wait_time)
                    continue
            elif e.status >= 500: 
                if attempt < retries:
                    await asyncio.sleep(0.5)
                    continue
            return None
        except (aiohttp.ClientError, IOError, Exception):
            if attempt < retries:
                await asyncio.sleep(0.3)
                continue
            return None
    return None

async def make_pano(image_id: str, pano_width: int, pano_height: int, tile_width: int, tile_height: int, auto_height: bool, zoom: int, filename: str = "pano.jpg") -> None:
    x_range = math.ceil(pano_width / tile_width)
    y_range = math.ceil(pano_height / tile_height)
    total_tiles = x_range * y_range
    
    if auto_height and pano_height != int(pano_width / 2):
        pano_height = int(pano_width / 2)

    pano = Image.new("RGB", (pano_width, pano_height))
    print(f"Total tiles to process: {total_tiles}")

    bar = progressbar.ProgressBar(max_value=total_tiles)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Referer': 'https://yandex.ru/maps/',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
    }

    semaphore = asyncio.Semaphore(10)  
    
    async def fetch_tile_with_limit(session, url):
        async with semaphore:
            await asyncio.sleep(0.1) 
            return await fetch_tile(session, url)
    
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for x in range(x_range):
            for y in range(y_range):
                tile_url = f"https://pano.maps.yandex.net/{image_id}/{zoom}.{x}.{y}"
                tasks.append(fetch_tile_with_limit(session, tile_url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        tile_index = 0
        loaded_tiles = 0
        failed_tiles = 0
        black_tiles = 0
        for x in range(x_range):
            for y in range(y_range):
                tile = results[tile_index]
                tile_index += 1
                if isinstance(tile, Exception):
                    tile = None
                
                if tile:
                    if tile.mode == 'RGB':
                        pixels = list(tile.getdata())
                        if len(pixels) > 0:
                            black_pixels = sum(1 for p in pixels if p == (0, 0, 0))

                            if black_pixels == len(pixels):
                                black_tiles += 1
                                failed_tiles += 1
                                continue
                    
                    pano.paste(tile, (x * tile_width, y * tile_height))
                    loaded_tiles += 1
                    bar.update(bar.value + 1)
                else:
                    failed_tiles += 1
        
        print(f"\nLoaded tiles: {loaded_tiles}/{total_tiles} ({loaded_tiles*100//total_tiles if total_tiles > 0 else 0}%)")
        if failed_tiles > 0:
            print(f"Failed tiles: {failed_tiles} ({failed_tiles*100//total_tiles if total_tiles > 0 else 0}%)")
            if black_tiles > 0:
                print(f"  - Completely black tiles (rejected): {black_tiles}")
            if failed_tiles > total_tiles * 0.3:
                print("Warning: More than 30% of tiles failed to load or were completely black.")
                print("   This could be due to:")
                print("   - Network issues or rate limiting")
                print("   - Some tiles are not available on Yandex servers")
                print("   - Panorama taken at night or in very dark area")
                print("   - Try using a different zoom level (-z 0 or -z 2)")

    pano.save(filename)
    print(f"Panorama saved as {filename}")

def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Download and assemble a Yandex panorama.")
    parser.add_argument("-c", "--coordinates", required=True, help="Coordinates in the format 'latitude,longitude'")
    parser.add_argument("-z", "--zoom", type=int, default=0, help="Zoom level (default: 0)")
    parser.add_argument("-o", "--output", default="pano.jpg", help="Output file name (default: pano.jpg)")
    parser.add_argument("-a", "--auto-height", action="store_true", help="Automatically adjust panorama height")

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    try:
        lat, lon = map(float, args.coordinates.split(","))
    except ValueError:
        print("Invalid coordinates. Please provide in the format 'latitude,longitude'")
        sys.exit(1)

    api_url = (
        f"https://api-maps.yandex.ru/services/panoramas/1.x/?l=stv&lang=ru_RU&ll="
        f"{lon},{lat}&origin=userAction&provider=streetview"
    )

    data = fetch_and_process_json(api_url)
    if not data:
        sys.exit(1)

    if data.get("status") == "error":
        error_code = data.get("code", "unknown")
        if error_code == 404:
            print(f"Panorama not found at coordinates {lat},{lon}. Please check if there is a street view panorama at this location.")
        else:
            print(f"API error: {error_code}")
        sys.exit(1)

    try:
        pano_data = data["data"]["Data"]
        images = pano_data["Images"]
        image_id = images["imageId"]
        tile_width = images["Tiles"]["width"]
        tile_height = images["Tiles"]["height"]

        timestamp = pano_data.get("timestamp")
        if timestamp:
            capture_date = datetime.fromtimestamp(timestamp)
            print(f"Дата съемки: {capture_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Год: {capture_date.year}, Месяц: {capture_date.month}, День: {capture_date.day}")
        else:
            print("Warning: Capture date information unavailable")
        
        zooms_list = images.get("Zooms", [])
        zoom_data = None
        for zoom_item in zooms_list:
            if zoom_item.get("level") == args.zoom:
                zoom_data = zoom_item
                break
        
        if zoom_data is None:
            available_zooms = [z.get("level") for z in zooms_list]
            print(f"Zoom level {args.zoom} is not available. Available zoom levels: {available_zooms}")
            sys.exit(1)
        
        pano_width = zoom_data["width"]
        pano_height = zoom_data["height"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"Invalid panorama data structure: {e}")
        print("The API response may have changed or the data format is unexpected.")
        sys.exit(1)

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) # to work properly on windows

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            make_pano(image_id, pano_width, pano_height, tile_width, tile_height, args.auto_height, args.zoom, args.output)
        )
    finally:
        loop.close()

if __name__ == "__main__":
    main()
