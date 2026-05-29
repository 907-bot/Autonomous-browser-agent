import { create } from 'zustand'
import { TaskHistoryItem, AgentStatus } from '@/types'

interface AgentState {
  // Task execution
  isExecuting: boolean
  currentTask: string
  currentUrl: string
  status: string
  results: string
  
  // Settings
  headless: boolean
  maxSteps: number
  
  // History
  taskHistory: TaskHistoryItem[]
  
  // Agent status
  agentStatus: AgentStatus
  
  // Decomposition
  decomposition: string
  isDecomposing: boolean
  
  // Actions
  setExecuting: (executing: boolean) => void
  setCurrentTask: (task: string) => void
  setCurrentUrl: (url: string) => void
  setStatus: (status: string) => void
  setResults: (results: string) => void
  setHeadless: (headless: boolean) => void
  setMaxSteps: (maxSteps: number) => void
  addToHistory: (item: TaskHistoryItem) => void
  clearHistory: () => void
  setAgentStatus: (status: AgentStatus) => void
  setDecomposition: (decomposition: string) => void
  setDecomposing: (decomposing: boolean) => void
  reset: () => void
}

const initialState = {
  isExecuting: false,
  currentTask: '',
  currentUrl: 'https://www.google.com',
  status: '',
  results: '',
  headless: true,
  maxSteps: 30,
  taskHistory: [],
  agentStatus: {
    status: 'idle' as const,
    active_task: null,
    queue_length: 0,
    last_action: '',
  },
  decomposition: '',
  isDecomposing: false,
}

export const useAgentStore = create<AgentState>((set) => ({
  ...initialState,
  
  setExecuting: (executing) => set({ isExecuting: executing }),
  setCurrentTask: (task) => set({ currentTask: task }),
  setCurrentUrl: (url) => set({ currentUrl: url }),
  setStatus: (status) => set({ status }),
  setResults: (results) => set({ results }),
  setHeadless: (headless) => set({ headless }),
  setMaxSteps: (maxSteps) => set({ maxSteps }),
  
  addToHistory: (item) => set((state) => ({
    taskHistory: [item, ...state.taskHistory].slice(0, 10),
  })),
  
  clearHistory: () => set({ taskHistory: [] }),
  setAgentStatus: (agentStatus) => set({ agentStatus }),
  setDecomposition: (decomposition) => set({ decomposition }),
  setDecomposing: (decomposing) => set({ isDecomposing: decomposing }),
  reset: () => set(initialState),
}))