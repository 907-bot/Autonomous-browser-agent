export interface TaskResult {
  success: boolean;
  steps: Step[];
  sub_tasks: string[];
  error?: string;
}

export interface Step {
  action: string;
  description: string;
  success: boolean;
  timestamp: string;
}

export interface TaskHistoryItem {
  id: string;
  task: string;
  url: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  success: boolean;
  steps_completed: number;
  timestamp: string;
}

export interface AgentStatus {
  status: 'idle' | 'monitoring' | 'running' | 'completed' | 'error';
  active_task: string | null;
  queue_length: number;
  last_action: string;
  last_update?: string;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  success: boolean;
}