import aiohttp
import asyncio
import time
import numpy as np
import aiofiles


# Асинхронная функция для отправки запроса с картинкой
async def send_image_request(session, url, image_path):
    async with aiofiles.open(image_path, "rb") as image_file:
        start_time = time.time()
        image_data = await image_file.read()
        form = aiohttp.FormData()
        form.add_field(
            "image", image_data, filename="test_image.jpg", content_type="image/jpeg"
        )
        async with session.post(url, data=form) as response:
            return response.status, time.time() - start_time


# Асинхронная функция для проверки нагрузки
async def load_test(url, image_path, duration=60):
    start_time = time.time()
    request_count = 0
    ok_count = 0
    iter_duration_list = []

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration:
            status, iter_duration = await send_image_request(session, url, image_path)
            iter_duration_list.append(iter_duration)
            if status == 200:
                ok_count += 1
            request_count += 1

    print(f"Total requests made: {request_count}")
    print(f"Successful requests made: {ok_count}")
    print(f"Average response time, s: {np.mean(iter_duration_list)}")


if __name__ == "__main__":
    # Запуск асинхронной функции
    asyncio.run(
        load_test("http://localhost/find_simular_images/", "tests/test_image.jpg")
    )
