const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ExecuteTaskParams {
  task: string;
  url: string;
  headless: boolean;
  max_steps: number;
}

interface TaskResponse {
  success: boolean;
  status_text: string;
  results_json: string;
  history_text: string;
}

interface DecomposeResponse {
  decomposition: string;
  error?: string;
}

interface StatusResponse {
  status: string;
  active_task: string | null;
  queue_length: number;
  history_count: number;
  last_action: string;
  last_update: string;
}

export async function executeTask(params: ExecuteTaskParams): Promise<TaskResponse> {
  const response = await fetch(`${API_BASE_URL}/execute`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function decomposeTask(task: string): Promise<DecomposeResponse> {
  const response = await fetch(`${API_BASE_URL}/decompose`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ task }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function getAgentStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE_URL}/status`);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function getHealth(): Promise<{ status: string; agent_available: boolean }> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}