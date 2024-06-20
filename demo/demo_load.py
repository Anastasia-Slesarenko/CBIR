import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import asyncio
import time
import aiohttp
import numpy as np
import random
from lib.settings import VM_IP


async def send_image_request(session, url, image_path):
    # Open the file and read it
    with open(image_path, "rb") as f:
        file_data = f.read()
    # Create FormData and add the file data
    data = aiohttp.FormData()
    data.add_field(
        "image",
        file_data,
        filename=image_path,
        content_type="image/jpeg",
    )
    start_time = time.time()
    async with session.post(url, data=data) as response:
        return response.status, time.time() - start_time


async def load_test(url, image_folder, duration=60):
    start_time = time.time()
    request_count = 0
    ok_count = 0
    iter_duration_list = []

    # Get list of image files in the specified folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
    ]

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration:
            # Choose a random image from the folder
            image_path = os.path.join(
                image_folder, random.choice(image_files)
            )
            status, iter_duration = await send_image_request(
                session, url, image_path
            )
            iter_duration_list.append(iter_duration)
            if status == 200:
                ok_count += 1
            request_count += 1

    total_time_elapsed = time.time() - start_time
    print(f"Total requests made: {request_count}")
    print(f"Successful requests made: {ok_count}")
    print(f"Average response time, s: {np.mean(iter_duration_list)}")

    perc_95 = np.percentile(iter_duration_list, 95)
    print(f"95th percentile response time, s: {perc_95}")

    rps = request_count / total_time_elapsed
    print(f"Requests Per Second (RPS): {rps:.2f}")


if __name__ == "__main__":
    asyncio.run(
        load_test(
            f"http://{VM_IP}/find_similar_image_urls_by_image",
            "demo/images",  # Specify the folder containing images
        )
    )
