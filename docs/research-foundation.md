# Research foundation operations

Research runs use the database owner connection and keep their own run, dossier, model-call, and
tool-call records. The API service does not install or import the research package.

LangGraph's Postgres checkpointer owns its schema and versioning. Each `start` or `continue` job
calls `PostgresSaver.setup()` before loading the graph; the operation is idempotent. Migration
`0016_research` therefore creates the research records but does not create or drop LangGraph's
checkpoint tables. The checkpointer receives a plain psycopg URL derived from
`OWNER_DATABASE_URL` by removing SQLAlchemy's `+psycopg` driver suffix.

To pause between jobs, the graph interrupts after `screen`. The initial `start` command records the
run as waiting. A later `continue --run-id` loads the same `thread_id` from Postgres and resumes
from that checkpoint.
