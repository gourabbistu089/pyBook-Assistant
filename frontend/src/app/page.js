'use client'
import React, { useState, useRef, useEffect } from 'react';
import { Send, Code2, Loader2, Copy, Check, Plus, MessageSquare, Trash2, Menu, X, History, Terminal, Sparkles } from 'lucide-react';

export default function PyHelperRAG() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hi! I\'m PyHelper, your Python programming assistant. What would you like to learn today?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [copiedCodeIndex, setCopiedCodeIndex] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState([
    { id: 1, title: 'Python Basics', date: 'Today', active: true },
    { id: 2, title: 'List Comprehensions', date: 'Yesterday', active: false },
    { id: 3, title: 'Decorators Explained', date: '2 days ago', active: false },
  ]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleCopy = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleCopyCode = (code, index) => {
    navigator.clipboard.writeText(code);
    setCopiedCodeIndex(index);
    setTimeout(() => setCopiedCodeIndex(null), 2000);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input })
      });

      if (!response.ok) throw new Error('Failed to get response');

      const data = await response.json();
      
      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: 'I\'m having trouble connecting to the server. Please make sure the backend is running on http://localhost:8000',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const exampleQuestions = [
    "How do list comprehensions work?",
    "Explain Python decorators",
    "What are generators used for?",
    "Best practices for error handling"
  ];

  const handleExampleClick = (question) => {
    setInput(question);
    inputRef.current?.focus();
  };

  const handleNewChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'Hi! I\'m PyHelper, your Python programming assistant. I can help you understand Python concepts, debug code, and answer your questions. What would you like to learn today?',
        timestamp: new Date()
      }
    ]);
  };

  // Enhanced markdown parser with code block support
  const parseMarkdown = (text) => {
    const parts = [];
    let currentIndex = 0;
    
    const codeBlockRegex = /```(?:python)?\n?([\s\S]*?)```|`([^`]+)`/g;
    let match;
    
    while ((match = codeBlockRegex.exec(text)) !== null) {
      if (match.index > currentIndex) {
        parts.push({
          type: 'text',
          content: text.substring(currentIndex, match.index)
        });
      }
      
      if (match[1] !== undefined) {
        parts.push({
          type: 'codeBlock',
          content: match[1].trim()
        });
      } else if (match[2] !== undefined) {
        parts.push({
          type: 'inlineCode',
          content: match[2]
        });
      }
      
      currentIndex = match.index + match[0].length;
    }
    
    if (currentIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.substring(currentIndex)
      });
    }
    
    return parts;
  };

  const formatText = (text) => {
    const lines = text.split('\n');
    
    return lines.map((line, lineIndex) => {
      const listMatch = line.match(/^[\s]*[*-]\s+(.+)$/);
      const numberedListMatch = line.match(/^[\s]*\d+\.\s+(.+)$/);
      
      if (listMatch) {
        return (
          <div key={lineIndex} className="flex gap-2.5 my-1.5">
            <span className="text-blue-500 mt-1 text-xs font-bold">•</span>
            <span className="flex-1">{formatInlineStyles(listMatch[1])}</span>
          </div>
        );
      }
      
      if (numberedListMatch) {
        return (
          <div key={lineIndex} className="flex gap-2.5 my-1.5">
            <span className="text-blue-500 min-w-[1.25rem] text-xs font-semibold">{line.match(/^\s*(\d+)\./)[1]}.</span>
            <span className="flex-1">{formatInlineStyles(numberedListMatch[1])}</span>
          </div>
        );
      }
      
      if (line.trim()) {
        return <div key={lineIndex} className="my-1.5">{formatInlineStyles(line)}</div>;
      }
      
      return <div key={lineIndex} className="h-2"></div>;
    });
  };

  const formatInlineStyles = (text) => {
    const parts = [];
    let currentIndex = 0;
    
    const styleRegex = /\*\*(.+?)\*\*|\*(.+?)\*/g;
    let match;
    
    while ((match = styleRegex.exec(text)) !== null) {
      if (match.index > currentIndex) {
        parts.push(text.substring(currentIndex, match.index));
      }
      
      if (match[1] !== undefined) {
        parts.push(
          <strong key={match.index} className="font-semibold text-gray-900">
            {match[1]}
          </strong>
        );
      } else if (match[2] !== undefined) {
        parts.push(
          <em key={match.index} className="italic">
            {match[2]}
          </em>
        );
      }
      
      currentIndex = match.index + match[0].length;
    }
    
    if (currentIndex < text.length) {
      parts.push(text.substring(currentIndex));
    }
    
    return parts.length > 0 ? parts : text;
  };

  const renderMessageContent = (content, messageIndex) => {
    const parts = parseMarkdown(content);
    
    return parts.map((part, index) => {
      if (part.type === 'codeBlock') {
        const codeKey = `${messageIndex}-${index}`;
        return (
          <div key={index} className="my-4 rounded-2xl overflow-hidden border border-gray-200 bg-gradient-to-br from-gray-50 to-gray-100 shadow-lg">
            <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-gray-800 to-gray-900 border-b border-gray-700">
              <div className="flex items-center gap-2">
                <Terminal className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-xs font-semibold text-gray-300">Python</span>
              </div>
              <button
                onClick={() => handleCopyCode(part.content, codeKey)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-all duration-200"
              >
                {copiedCodeIndex === codeKey ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-green-400" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy code</span>
                  </>
                )}
              </button>
            </div>
            <div className="p-5 overflow-x-auto">
              <pre className="text-sm font-mono leading-relaxed">
                <code className="text-gray-800">{part.content}</code>
              </pre>
            </div>
          </div>
        );
      } else if (part.type === 'inlineCode') {
        return (
          <code key={index} className="px-2 py-0.5 mx-0.5 bg-blue-100 rounded-md text-blue-800 font-mono text-[13px] border border-blue-200">
            {part.content}
          </code>
        );
      } else {
        return (
          <div key={index} className="text-[15px] leading-relaxed">
            {formatText(part.content)}
          </div>
        );
      }
    });
  };

  return (
    <div className="h-screen bg-white text-gray-800 flex overflow-hidden">
      {/* Sidebar */}
      <aside 
        className={`${
          sidebarOpen ? 'w-72' : 'w-0'
        } bg-gray-50 backdrop-blur-xl border-r border-gray-200 flex flex-col transition-all duration-300 ease-out overflow-hidden shadow-sm`}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2.5 px-4 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl transition-all duration-200 text-sm font-semibold shadow-lg hover:shadow-xl hover:scale-[1.02]"
          >
            <Plus className="w-4 h-4" />
            <span>New conversation</span>
          </button>
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto py-3 px-3 space-y-1">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              className={`w-full flex items-center gap-3 px-3.5 py-3 rounded-xl transition-all duration-200 text-left group ${
                conv.active 
                  ? 'bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-900 shadow-md' 
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
              }`}
            >
              <MessageSquare className={`w-4 h-4 flex-shrink-0 ${conv.active ? 'text-blue-600' : 'text-gray-400 group-hover:text-gray-600'}`} />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{conv.title}</p>
                <p className="text-xs text-gray-500 mt-0.5">{conv.date}</p>
              </div>
            </button>
          ))}
        </div>

        {/* Sidebar Footer */}
        <div className="p-3 border-t border-gray-200 space-y-1">
          <button className="w-full flex items-center gap-3 px-3.5 py-2.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-xl transition-all duration-200 text-sm">
            <History className="w-4 h-4" />
            <span>History</span>
          </button>
          <button className="w-full flex items-center gap-3 px-3.5 py-2.5 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-xl transition-all duration-200 text-sm">
            <Trash2 className="w-4 h-4" />
            <span>Clear conversations</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <header className="h-16 border-b border-gray-200 flex items-center justify-between px-5 bg-gray-50 backdrop-blur-xl shadow-sm">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-xl transition-all duration-200"
            >
              {sidebarOpen ? <X className="w-5 h-5 text-gray-600" /> : <Menu className="w-5 h-5 text-gray-600" />}
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
                <Code2 className="w-5 h-5 text-white" strokeWidth={2.5} />
              </div>
              <div>
                <h1 className="text-base font-bold text-gray-900">PyHelper</h1>
                <p className="text-xs text-gray-500">Python programming assistant</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-green-100 border border-green-300 rounded-full">
            <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-xs font-semibold text-green-700">Ready</span>
          </div>
        </header>

        {/* Chat Content */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-4xl mx-auto">
            {messages.length === 1 && (
              <div className="text-center py-24 space-y-10">
                <div className="relative inline-flex">
                  <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-3xl flex items-center justify-center shadow-2xl">
                    <Code2 className="w-10 h-10 text-white" strokeWidth={2} />
                  </div>
                  <div className="absolute -top-2 -right-2 w-6 h-6 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-full flex items-center justify-center shadow-lg">
                    <Sparkles className="w-3.5 h-3.5 text-white" />
                  </div>
                </div>
                
                <div className="space-y-3">
                  <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    Welcome to PyHelper
                  </h2>
                  <p className="text-gray-600 text-base max-w-lg mx-auto leading-relaxed">
                    Your intelligent Python programming assistant. Ask me anything about Python and I'll provide detailed, helpful explanations.
                  </p>
                </div>

                <div className="space-y-4 pt-8">
                  <p className="text-sm text-gray-500 font-semibold">Try asking about</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto">
                    {exampleQuestions.map((question, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleExampleClick(question)}
                        className="p-4 bg-white hover:bg-gradient-to-br hover:from-blue-50 hover:to-indigo-50 border border-gray-200 hover:border-blue-300 rounded-2xl transition-all duration-200 text-left text-sm text-gray-700 hover:text-gray-900 shadow-sm hover:shadow-xl group"
                      >
                        <span className="block font-medium">{question}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-8">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                >
                  {/* Avatar */}
                  <div className="flex-shrink-0 mt-1">
                    {message.role === 'assistant' ? (
                      <div className="w-9 h-9 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg">
                        <Code2 className="w-5 h-5 text-white" strokeWidth={2.5} />
                      </div>
                    ) : (
                      <div className="w-9 h-9 rounded-2xl bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center font-semibold text-sm text-white shadow-lg">
                        You
                      </div>
                    )}
                  </div>

                  {/* Message Content */}
                  <div className="flex-1 min-w-0 max-w-3xl">
                    <div
                      className={`rounded-2xl px-5 py-4 ${
                        message.role === 'user'
                          ? 'bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-lg'
                          : message.isError
                          ? 'bg-red-50 border border-red-200 text-red-800'
                          : 'bg-white text-gray-700 border border-gray-200 shadow-md'
                      }`}
                    >
                      {renderMessageContent(message.content, index)}
                    </div>

                    {message.role === 'assistant' && !message.isError && (
                      <div className="flex items-center gap-4 mt-2.5 px-2">
                        <button
                          onClick={() => handleCopy(message.content, index)}
                          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-blue-600 transition-colors"
                        >
                          {copiedIndex === index ? (
                            <>
                              <Check className="w-3.5 h-3.5 text-green-600" />
                              <span className="text-green-600">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" />
                              <span>Copy response</span>
                            </>
                          )}
                        </button>
                        <span className="text-xs text-gray-400">
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-4">
                  <div className="flex-shrink-0 mt-1">
                    <div className="w-9 h-9 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg">
                      <Code2 className="w-5 h-5 text-white" strokeWidth={2.5} />
                    </div>
                  </div>
                  <div className="flex-1 max-w-3xl">
                    <div className="bg-white border border-gray-200 rounded-2xl px-5 py-4 shadow-md">
                      <div className="flex items-center gap-3">
                        <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                        <span className="text-sm text-gray-600">Thinking...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 bg-gray-50 backdrop-blur-xl px-4 py-5 shadow-sm">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="flex items-end gap-3 bg-white border-2 border-gray-200 focus-within:border-blue-500 rounded-2xl p-2 transition-all duration-200 shadow-lg">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about Python programming..."
                className="flex-1 bg-transparent px-4 py-3 text-gray-900 placeholder-gray-400 focus:outline-none text-[15px] resize-none"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed p-3 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl hover:scale-105"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin text-white" />
                ) : (
                  <Send className="w-5 h-5 text-white" strokeWidth={2} />
                )}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-3 text-center">
              PyHelper can make mistakes. Always verify important information.
            </p>
          </form>
        </div>
      </main>

      <style jsx>{`
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
          width: 10px;
          height: 10px;
        }

        ::-webkit-scrollbar-track {
          background: transparent;
        }

        ::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, rgb(59, 130, 246), rgb(99, 102, 241));
          border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, rgb(37, 99, 235), rgb(79, 70, 229));
        }

        /* Smooth transitions */
        * {
          scroll-behavior: smooth;
        }
      `}</style>
    </div>
  );
}