import asyncio
import time
import aiohttp
import numpy as np


async def send_image_request(session, url, image_path):
    async with aiohttp.ClientSession() as session:
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


async def load_test(url, image_path, duration=60):
    start_time = time.time()
    request_count = 0
    ok_count = 0
    iter_duration_list = []

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration:
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

    rps = request_count / total_time_elapsed
    print(f"Requests Per Second (RPS): {rps:.2f}")


if __name__ == "__main__":

    asyncio.run(
        load_test(
            "http://158.160.64.37/find_simular_images",
            "tests/query_image_test.jpg",
        )
    )

# 20k images
# vits16, emb_size: 384
# Total requests made: 176
# Successful requests made: 176
# Average response time, s: 0.34
# Requests Per Second (RPS): 2.93

# vitb32_unicom, emb_size: 512,
# Total requests made: 189
# Successful requests made: 189
# Average response time, s: 0.32
# Requests Per Second (RPS): 3.15

# vitl14_336px_unicom, emb_size: 768
# Total requests made: 43
# Successful requests made: 43
# Average response time, s: 1.4
# Requests Per Second (RPS): 0.71
