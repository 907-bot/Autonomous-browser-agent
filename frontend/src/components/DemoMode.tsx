'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Bot, Loader2 } from 'lucide-react'

interface DemoResponse {
  success: boolean
  status_text: string
  results_json: string
}

export default function DemoMode() {
  const [isLoading, setIsLoading] = useState(false)
  const [response, setResponse] = useState<DemoResponse | null>(null)

  const simulateTask = async () => {
    setIsLoading(true)
    setResponse(null)

    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 2000))

    const demoResponse: DemoResponse = {
      success: true,
      status_text: `✅ Demo Mode

📋 Task: Sample task executed
🌐 URL: https://example.com
📊 Steps Completed: 5/10
✅ Successful Steps: 4
⏱️ Timestamp: ${new Date().toLocaleString()}

**Sub-tasks:** 3
• Navigate to the website
• Fill out the form
• Submit the data`,
      results_json: JSON.stringify({
        success: true,
        demo: true,
        message: "This is a demo response. Connect to the backend for real functionality.",
        timestamp: new Date().toISOString(),
        features_available: [
          "Task execution",
          "Task decomposition",
          "Status monitoring",
          "History tracking"
        ]
      }, null, 2)
    }

    setResponse(demoResponse)
    setIsLoading(false)
  }

  return (
    <div className="min-h-screen bg-dark-500 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-md w-full bg-dark-300/50 backdrop-blur-sm rounded-2xl border border-white/10 p-8 text-center"
      >
        <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center">
          <Bot className="w-10 h-10 text-cyan-400" />
        </div>

        <h1 className="text-2xl font-bold text-white mb-2">Demo Mode</h1>
        <p className="text-gray-400 mb-6">
          The backend is not available. This is a demo of the frontend interface.
        </p>

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={simulateTask}
          disabled={isLoading}
          className="w-full py-3 px-6 bg-gradient-to-r from-cyan-600 to-purple-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-cyan-500/25 transition-all flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Simulating...
            </>
          ) : (
            <>
              <Bot className="w-5 h-5" />
              Run Demo Task
            </>
          )}
        </motion.button>

        {response && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 text-left"
          >
            <h3 className="text-sm font-medium text-gray-300 mb-2">Status</h3>
            <pre className="bg-dark-400/50 rounded-lg p-4 text-sm text-gray-300 whitespace-pre-wrap">
              {response.status_text}
            </pre>

            <h3 className="text-sm font-medium text-gray-300 mb-2 mt-4">Results (JSON)</h3>
            <pre className="bg-dark-400/50 rounded-lg p-4 text-xs text-gray-400 max-h-48 overflow-auto">
              {response.results_json}
            </pre>
          </motion.div>
        )}
      </motion.div>
    </div>
  )
}