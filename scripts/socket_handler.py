# socket_handler.py
import asyncio
import websockets

async def handle_connection(websocket):
    print("Client connected:", websocket.remote_address)

    try:
        async for message in websocket:
            print(f"Received from client: {message}")

            if message == "connect_client":
                await websocket.send("Connection acknowledged")
            else:
                print("Received form data:", message)
                await websocket.send("Form data received and processed")

    except websockets.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
