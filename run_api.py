import multiprocessing as mp

def main():
    # Critical for macOS + PyTorch
    mp.set_start_method("spawn", force=True)

    import uvicorn
    uvicorn.run(
        "app.api:app",
        host="127.0.0.1",
        port=8000,
        workers=1,
        reload=False,
        loop="asyncio",
        http="h11"
    )

if __name__ == "__main__":
    main()

