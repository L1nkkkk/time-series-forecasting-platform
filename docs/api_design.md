# API Design

The MVP API is intentionally synchronous. It is designed to make later async job
execution straightforward without putting queue details into the trainer.

## Endpoints

### GET /health

Returns service health and version.

### GET /datasets

Returns registered datasets and catalog metadata.

### GET /models

Returns registered model names.

### GET /experiments

Returns local experiment summaries discovered under the configured run root.

### POST /experiments/train

Accepts a validated config payload and runs a synchronous demo training job.
Future iterations should move this endpoint to a durable background worker.

## Error Handling

- Invalid configs return HTTP 422 through Pydantic validation.
- Runtime training failures should return clear HTTP 500 responses with a
  concise message and server-side logs.
