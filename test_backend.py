
import httpx
import asyncio

async def test_step2(image_path, option_json):
    url = "http://127.0.0.1:8000/step2"
    with open(option_json) as f:
        options = f.read()

    files = {'image': open(image_path, 'rb')}
    data = {'option_json': options}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, files=files, data=data)
        
        if response.status_code == 200:
            print("Step 2 Response Received:")
            return response.content
        else:
            print(f"Step 2 failed with status code: {response.status_code}")

async def test_step4(image_path):
    url = "http://127.0.0.1:8000/step4"
    
    files = {'image': open(image_path, 'rb')}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, files=files)
        
        if response.status_code == 200:
            print("Step 4 Response Received:")
            return response.content
        else:
            print(f"Step 4 failed with status code: {response.status_code}")

async def main():
    # Update these paths to point to your actual image and JSON option files
    image_path = "path/to/your/image.png"
    option_json = "path/to/your/options.json"
    
    step2_result = await test_step2(image_path, option_json)
    step4_result = await test_step4(image_path)

    # You can further process or validate the results here
    if step2_result:
        with open("step2_output.png", "wb") as f:
            f.write(step2_result)

    if step4_result:
        with open("step4_output.png", "wb") as f:
            f.write(step4_result)

if __name__ == "__main__":
    asyncio.run(main())
