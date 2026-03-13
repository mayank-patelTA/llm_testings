from langsmith import Client

client = Client()

run = client.create_run(
    name="test",
    run_type="chain",
    inputs={"hello": "world"}
)

print(run)