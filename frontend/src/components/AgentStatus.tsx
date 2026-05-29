'use client'

import { motion } from 'framer-motion'
import { Activity, Server, Clock, Zap } from 'lucide-react'
import { useAgentStore } from '@/lib/store'
import { cn } from '@/lib/utils'

const statusConfig = {
  idle: { color: 'gray', icon: '○', label: 'Idle' },
  monitoring: { color: 'cyan', icon: '⚡', label: 'Monitoring' },
  running: { color: 'amber', icon: '⟳', label: 'Running' },
  completed: { color: 'emerald', icon: '✓', label: 'Completed' },
  error: { color: 'rose', icon: '✕', label: 'Error' },
}

export default function AgentStatus() {
  const { agentStatus } = useAgentStore()
  
  const config = statusConfig[agentStatus.status as keyof typeof statusConfig] || statusConfig.idle

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Activity className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold text-white">Agent Status</h3>
      </div>

      {/* Main Status Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-dark-300 to-dark-400 rounded-xl border border-white/5 p-6"
      >
        <div className="flex items-center gap-4">
          {/* Status Indicator */}
          <div className="relative">
            <div className={cn(
              'w-16 h-16 rounded-full flex items-center justify-center',
              `bg-${config.color}-500/20`
            )}>
              <motion.span
                animate={{ 
                  scale: agentStatus.status === 'running' ? [1, 1.1, 1] : 1,
                }}
                transition={{ duration: 1, repeat: Infinity }}
                className="text-2xl"
              >
                {config.icon}
              </motion.span>
            </div>
            {/* Pulse effect for running state */}
            {agentStatus.status === 'running' && (
              <motion.div
                animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="absolute inset-0 rounded-full border-2 border-cyan-500"
              />
            )}
          </div>

          {/* Status Text */}
          <div>
            <h4 className={cn(
              'text-xl font-bold',
              `text-${config.color}-400`
            )}>
              {config.label}
            </h4>
            {agentStatus.active_task && (
              <p className="text-sm text-gray-400 mt-1">
                Current: {agentStatus.active_task}
              </p>
            )}
          </div>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Queue Length */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Server className="w-4 h-4 text-purple-400" />
            <span className="text-xs text-gray-400">Queue</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {agentStatus.queue_length}
          </div>
          <div className="text-xs text-gray-500">pending tasks</div>
        </motion.div>

        {/* Last Update */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-amber-400" />
            <span className="text-xs text-gray-400">Last Update</span>
          </div>
          <div className="text-lg font-bold text-white truncate">
            {agentStatus.last_update 
              ? new Date(agentStatus.last_update).toLocaleTimeString()
              : 'N/A'}
          </div>
          <div className="text-xs text-gray-500">HH:MM:SS</div>
        </motion.div>
      </div>

      {/* Last Action */}
      {agentStatus.last_action && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-emerald-400" />
            <span className="text-xs text-gray-400">Last Action</span>
          </div>
          <p className="text-sm text-gray-300 truncate">
            {agentStatus.last_action}
          </p>
        </motion.div>
      )}
    </div>
  )
}