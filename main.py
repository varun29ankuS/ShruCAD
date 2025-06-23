import React, { useState, useEffect } from 'react';

function App() {
  // Core state
  const [activeTab, setActiveTab] = useState('prompt');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImageUrl, setPreviewImageUrl] = useState(null);
  const [textPrompt, setTextPrompt] = useState('');
  const [complexity, setComplexity] = useState('medium');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [generatedScript, setGeneratedScript] = useState('');
  const [modelData, setModelData] = useState(null);
  const [progress, setProgress] = useState(0);
  
  // Drawing state
  const [cadFile, setCadFile] = useState(null);
  const [selectedViews, setSelectedViews] = useState(['front', 'side', 'top']);
  const [generatedDrawings, setGeneratedDrawings] = useState({});
  
  // Chat state
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState([{
    id: 1, type: 'assistant', content: "Hi! I can help with 3D modeling questions.", timestamp: new Date().toISOString()
  }]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);

  const backendUrl = 'https://forge-ai-backend.onrender.com';

  // Drawing views
  const drawingViews = [
    { id: 'front', name: 'Front View', icon: '⬜' },
    { id: 'side', name: 'Side View', icon: '📐' },
    { id: 'top', name: 'Top View', icon: '⬆️' },
    { id: 'isometric', name: 'Isometric', icon: '📦' }
  ];

  // Utility functions
  const simulateProgress = () => {
    setProgress(0);
    const interval = setInterval(() => {
      setProgress(prev => prev >= 90 ? (clearInterval(interval), 90) : prev + Math.random() * 15);
    }, 500);
    return interval;
  };

  const resetResults = () => {
    setMessage('');
    setGeneratedScript('');
    setModelData(null);
    setProgress(0);
  };

  // File handlers
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewImageUrl(URL.createObjectURL(file));
      resetResults();
    }
  };

  const handleCADFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setCadFile(file);
      setGeneratedDrawings({});
    }
  };

  // Generation functions
  const handleGenerateFromPrompt = async () => {
    if (!textPrompt.trim()) return;
    
    setIsLoading(true);
    setMessage('🤖 AI is interpreting your idea...');
    resetResults();
    const progressInterval = simulateProgress();
    
    try {
      const response = await fetch(`${backendUrl}/generate-from-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: textPrompt, complexity })
      });
      
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.detail || 'Generation failed');
      
      clearInterval(progressInterval);
      setProgress(100);
      setMessage('🎉 Your 3D model is ready!');
      setGeneratedScript(data.script);
      setModelData(data.stl_data);
      
    } catch (error) {
      clearInterval(progressInterval);
      setProgress(0);
      setMessage(`❌ Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateFromImage = async () => {
    if (!selectedFile) return;
    
    setIsLoading(true);
    setMessage('🔍 Analyzing image...');
    const progressInterval = simulateProgress();
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await fetch(`${backendUrl}/generate`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.detail || 'Generation failed');
      
      clearInterval(progressInterval);
      setProgress(100);
      setMessage('🎉 Model generation complete!');
      setGeneratedScript(data.script);
      setModelData(data.stl_data);
      
    } catch (error) {
      clearInterval(progressInterval);
      setProgress(0);
      setMessage(`❌ Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Drawing generation - UPDATED for backend compatibility
  const handleGenerateDrawings = async () => {
    if (!cadFile && !modelData) return;
    
    setIsLoading(true);
    setMessage('📐 Generating drawings...');
    const progressInterval = simulateProgress();
    
    try {
      // For now, simulate drawing generation since backend endpoint doesn't exist yet
      // You'll need to add this endpoint to your main.py:
      /*
      @app.post("/generate-drawings")
      async def generate_drawings(
          file: UploadFile = File(...),
          views: str = Form(...),
          scale: str = Form("1:1"),
          format: str = Form("svg")
      ):
          # Process STL file and generate technical drawings
          # Return base64 encoded drawing data for each view
          return {"drawings": {"front": "base64_data", "side": "base64_data", ...}}
      */
      
      // Temporary simulation - replace with actual API call
      setTimeout(() => {
        clearInterval(progressInterval);
        setProgress(100);
        setMessage('📐 Drawings generated!');
        
        // Simulate generated drawings
        const mockDrawings = {};
        selectedViews.forEach(view => {
          // Create a simple SVG placeholder
          const svg = `<svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
            <rect width="300" height="200" fill="#f8f9fa" stroke="#dee2e6"/>
            <text x="150" y="100" text-anchor="middle" font-family="Arial" font-size="16" fill="#6c757d">
              ${drawingViews.find(v => v.id === view)?.name || view} View
            </text>
            <text x="150" y="120" text-anchor="middle" font-family="Arial" font-size="12" fill="#6c757d">
              Generated from STL
            </text>
          </svg>`;
          mockDrawings[view] = btoa(svg);
        });
        
        setGeneratedDrawings(mockDrawings);
        setIsLoading(false);
      }, 2000);
      
    } catch (error) {
      clearInterval(progressInterval);
      setProgress(0);
      setMessage(`❌ Error: ${error.message}`);
      setIsLoading(false);
    }
  };

  // Chat functions
  const sendChatMessage = async () => {
    if (!chatInput.trim() || isChatLoading) return;
    
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: chatInput.trim(),
      timestamp: new Date().toISOString()
    };
    
    setChatMessages(prev => [...prev, userMessage]);
    const currentInput = chatInput.trim();
    setChatInput('');
    setIsChatLoading(true);
    
    try {
      // Prepare context about current work for better AI responses
      const context = {
        activeTab,
        hasModel: !!modelData,
        hasScript: !!generatedScript,
        hasDrawings: Object.keys(generatedDrawings).length > 0,
        currentPrompt: textPrompt,
        complexity,
        selectedViews,
        recentScript: generatedScript ? generatedScript.substring(0, 500) : null
      };
      
      const response = await fetch(`${backendUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: currentInput,
          context: context,
          conversation_history: chatMessages.slice(-6) // Last 6 messages for context
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Chat service temporarily unavailable');
      }
      
      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString()
      };
      
      setChatMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: `I'm having trouble connecting right now. Error: ${error.message}. Please try again in a moment.`,
        timestamp: new Date().toISOString()
      };
      
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const downloadSTL = () => {
    if (!modelData) return;
    
    try {
      const byteCharacters = atob(modelData);
      const byteNumbers = new Array(byteCharacters.length);
      
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = `model-${Date.now()}.stl`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
    } catch (error) {
      setMessage(`Download failed: ${error.message}`);
    }
  };

  const downloadDrawings = () => {
    Object.entries(generatedDrawings).forEach(([viewName, drawingData]) => {
      const link = document.createElement('a');
      link.href = `data:image/svg+xml;base64,${drawingData}`;
      link.download = `${viewName}-drawing.svg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  };

  const ModelPreview = ({ modelData }) => (
    <div style={{
      width: '100%', height: '250px', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderRadius: '8px', display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', color: '#fff', textAlign: 'center', padding: '20px'
    }}>
      <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🎯</div>
      <h3 style={{ margin: '0 0 8px 0', fontSize: '1.1rem' }}>3D Model Ready</h3>
      <p style={{ margin: 0, opacity: 0.8, fontSize: '0.9rem' }}>
        {modelData ? 'Generated successfully' : 'Ready for processing'}
      </p>
    </div>
  );

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      fontFamily: 'Arial, sans-serif', color: '#fff',
      marginRight: showChat ? '350px' : '0', transition: 'margin-right 0.3s ease'
    }}>
      
      {/* Header */}
      <header style={{
        background: 'rgba(255, 255, 255, 0.1)', padding: '20px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)', backdropFilter: 'blur(10px)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ margin: '0 0 5px 0', fontSize: '1.8rem' }}>🎯 RasterShape AI</h1>
            <p style={{ margin: 0, opacity: 0.8 }}>AI-powered 3D modeling and technical drawings</p>
          </div>
          <button 
            onClick={() => setShowChat(!showChat)}
            style={{
              background: showChat ? '#4facfe' : '#667eea', border: 'none', color: 'white',
              padding: '12px 20px', borderRadius: '8px', cursor: 'pointer', fontWeight: '600'
            }}
          >
            🤖 AI Chat
          </button>
        </div>
      </header>

      {/* Chat Panel */}
      <div style={{
        position: 'fixed', top: 0, right: showChat ? '0' : '-350px', width: '350px',
        height: '100vh', background: 'rgba(26, 26, 46, 0.95)', backdropFilter: 'blur(10px)',
        borderLeft: '1px solid rgba(255, 255, 255, 0.1)', display: 'flex', flexDirection: 'column',
        transition: 'right 0.3s ease', zIndex: 1000
      }}>
        <div style={{
          padding: '20px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center'
        }}>
          <h3 style={{ margin: 0, color: '#fff' }}>🤖 AI Assistant</h3>
          <button onClick={() => setShowChat(false)} style={{
            background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '1.2rem'
          }}>✕</button>
        </div>
        
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {chatMessages.map(msg => (
            <div key={msg.id} style={{
              display: 'flex', gap: '8px', alignItems: 'flex-start',
              flexDirection: msg.type === 'user' ? 'row-reverse' : 'row'
            }}>
              <div style={{
                width: '32px', height: '32px', borderRadius: '50%',
                background: msg.type === 'user' ? '#667eea' : 'rgba(255, 255, 255, 0.1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.9rem'
              }}>
                {msg.type === 'user' ? '👤' : '🤖'}
              </div>
              <div style={{
                background: msg.type === 'user' ? '#667eea' : 'rgba(255, 255, 255, 0.1)',
                padding: '10px 14px', borderRadius: '12px', maxWidth: '240px',
                fontSize: '0.9rem', lineHeight: 1.4
              }}>
                {msg.content}
              </div>
            </div>
          ))}
          {isChatLoading && (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', padding: '10px' }}>
              <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'rgba(255, 255, 255, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>🤖</div>
              <div style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.9rem' }}>Thinking...</div>
            </div>
          )}
        </div>
        
        <div style={{ padding: '20px', borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="text"
              placeholder="Ask about 3D modeling..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
              style={{
                flex: 1, background: 'rgba(255, 255, 255, 0.1)', border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '8px', padding: '10px', color: '#fff', fontSize: '0.9rem'
              }}
            />
            <button onClick={sendChatMessage} disabled={!chatInput.trim() || isChatLoading} style={{
              background: '#667eea', border: 'none', color: 'white', width: '40px', height: '40px',
              borderRadius: '8px', cursor: 'pointer', opacity: (!chatInput.trim() || isChatLoading) ? 0.5 : 1
            }}>🚀</button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main style={{ flex: 1, padding: '20px' }}>
        
        {/* Tabs */}
        <div style={{ marginBottom: '30px' }}>
          <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
            {[
              { id: 'prompt', icon: '✏️', label: 'Text to 3D' },
              { id: 'image', icon: '📷', label: 'Image to 3D' },
              { id: 'drawings', icon: '📐', label: '3D to Drawings' }
            ].map(tab => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                background: activeTab === tab.id ? 'rgba(255, 255, 255, 0.2)' : 'rgba(255, 255, 255, 0.1)',
                border: 'none', color: '#fff', padding: '12px 20px', borderRadius: '8px',
                cursor: 'pointer', fontWeight: '600', transition: 'all 0.3s ease'
              }}>
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'prompt' && (
          <div style={{ marginBottom: '30px' }}>
            <h3>🤖 Describe Your 3D Model</h3>
            <textarea
              placeholder="Example: A 50mm cube with a 20mm cylindrical hole through the center..."
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              disabled={isLoading}
              style={{
                width: '100%', height: '120px', background: 'rgba(255, 255, 255, 0.1)',
                border: '1px solid rgba(255, 255, 255, 0.2)', borderRadius: '8px',
                padding: '15px', color: '#fff', fontSize: '14px', resize: 'vertical'
              }}
            />
            <div style={{ display: 'flex', gap: '15px', marginTop: '15px', alignItems: 'center' }}>
              <select value={complexity} onChange={(e) => setComplexity(e.target.value)} disabled={isLoading} style={{
                background: 'rgba(255, 255, 255, 0.1)', border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '8px', padding: '10px', color: '#fff'
              }}>
                <option value="simple">Simple</option>
                <option value="medium">Medium</option>
                <option value="complex">Complex</option>
              </select>
              <button onClick={handleGenerateFromPrompt} disabled={isLoading || !textPrompt.trim()} style={{
                background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', border: 'none',
                color: 'white', padding: '12px 24px', borderRadius: '8px', cursor: 'pointer',
                fontWeight: '600', opacity: (isLoading || !textPrompt.trim()) ? 0.5 : 1
              }}>
                {isLoading ? '🔄 Generating...' : '🚀 Generate 3D Model'}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'image' && (
          <div style={{ marginBottom: '30px' }}>
            <h3>📷 Upload Technical Drawing</h3>
            <div style={{
              border: '2px dashed rgba(255, 255, 255, 0.3)', borderRadius: '8px',
              padding: '40px', textAlign: 'center', cursor: 'pointer'
            }} onClick={() => document.getElementById('file-input').click()}>
              <input id="file-input" type="file" onChange={handleFileChange} accept="image/*" disabled={isLoading} style={{ display: 'none' }} />
              {selectedFile ? (
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '10px' }}>📁</div>
                  <div>{selectedFile.name}</div>
                  <div style={{ fontSize: '0.9rem', opacity: 0.7, marginTop: '5px' }}>Click to change</div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '10px' }}>📤</div>
                  <div>Click to upload image</div>
                  <div style={{ fontSize: '0.9rem', opacity: 0.7, marginTop: '5px' }}>JPEG, PNG, BMP, TIFF</div>
                </div>
              )}
            </div>
            <button onClick={handleGenerateFromImage} disabled={isLoading || !selectedFile} style={{
              background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', border: 'none',
              color: 'white', padding: '12px 24px', borderRadius: '8px', cursor: 'pointer',
              fontWeight: '600', marginTop: '15px', opacity: (isLoading || !selectedFile) ? 0.5 : 1
            }}>
              {isLoading ? '🔄 Analyzing...' : '🚀 Generate from Image'}
            </button>
          </div>
        )}

        {activeTab === 'drawings' && (
          <div style={{ marginBottom: '30px' }}>
            <h3>📐 Generate Technical Drawings</h3>
            <div style={{
              border: '2px dashed rgba(255, 255, 255, 0.3)', borderRadius: '8px',
              padding: '40px', textAlign: 'center', cursor: 'pointer', marginBottom: '20px'
            }} onClick={() => document.getElementById('cad-file-input').click()}>
              <input id="cad-file-input" type="file" onChange={handleCADFileChange} accept=".stl" disabled={isLoading} style={{ display: 'none' }} />
              {cadFile ? (
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🎯</div>
                  <div>{cadFile.name}</div>
                  <div style={{ fontSize: '0.9rem', opacity: 0.7, marginTop: '5px' }}>STL file ready</div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '10px' }}>📤</div>
                  <div>Upload STL File</div>
                  {modelData && (
                    <button onClick={() => { setCadFile(null); }} style={{
                      background: 'rgba(255, 255, 255, 0.2)', border: 'none', color: '#fff',
                      padding: '8px 16px', borderRadius: '6px', cursor: 'pointer', marginTop: '10px'
                    }}>
                      📦 Use Current Model
                    </button>
                  )}
                </div>
              )}
            </div>
            
            <div style={{ marginBottom: '20px' }}>
              <h4>Select Views:</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
                {drawingViews.map(view => (
                  <div key={view.id} onClick={() => {
                    setSelectedViews(prev => 
                      prev.includes(view.id) ? prev.filter(id => id !== view.id) : [...prev, view.id]
                    );
                  }} style={{
                    background: selectedViews.includes(view.id) ? 'rgba(59, 130, 246, 0.3)' : 'rgba(255, 255, 255, 0.1)',
                    border: selectedViews.includes(view.id) ? '2px solid #3b82f6' : '2px solid transparent',
                    padding: '15px', borderRadius: '8px', cursor: 'pointer', textAlign: 'center', transition: 'all 0.3s ease'
                  }}>
                    <div style={{ fontSize: '1.5rem', marginBottom: '5px' }}>{view.icon}</div>
                    <div style={{ fontSize: '0.9rem' }}>{view.name}</div>
                  </div>
                ))}
              </div>
            </div>
            
            <button onClick={handleGenerateDrawings} disabled={isLoading || (!cadFile && !modelData) || selectedViews.length === 0} style={{
              background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', border: 'none',
              color: 'white', padding: '12px 24px', borderRadius: '8px', cursor: 'pointer',
              fontWeight: '600', opacity: (isLoading || (!cadFile && !modelData) || selectedViews.length === 0) ? 0.5 : 1
            }}>
              {isLoading ? '🔄 Generating...' : '📐 Generate Drawings'}
            </button>
          </div>
        )}

        {/* Progress */}
        {isLoading && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.1)', borderRadius: '8px', padding: '20px',
            marginBottom: '20px', textAlign: 'center'
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🤖</div>
            <div style={{
              background: 'rgba(255, 255, 255, 0.2)', borderRadius: '10px', height: '8px',
              marginBottom: '10px', overflow: 'hidden'
            }}>
              <div style={{
                background: 'linear-gradient(90deg, #4facfe, #00f2fe)', height: '100%',
                width: `${progress}%`, transition: 'width 0.3s ease'
              }} />
            </div>
            <div>{message}</div>
          </div>
        )}

        {/* Results */}
        {(generatedScript || modelData) && activeTab !== 'drawings' && (
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '20px', marginBottom: '20px'
          }}>
            <div style={{ background: 'rgba(255, 255, 255, 0.1)', borderRadius: '8px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                <h3 style={{ margin: 0 }}>⚙️ Generated Code</h3>
                {generatedScript && (
                  <button onClick={() => navigator.clipboard.writeText(generatedScript)} style={{
                    background: 'rgba(255, 255, 255, 0.2)', border: 'none', color: '#fff',
                    padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.8rem'
                  }}>📋 Copy</button>
                )}
              </div>
              <textarea value={generatedScript} readOnly style={{
                width: '100%', height: '200px', background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid rgba(255, 255, 255, 0.2)', borderRadius: '6px',
                padding: '10px', color: '#fff', fontSize: '12px', fontFamily: 'monospace'
              }} />
            </div>
            
            <div style={{ background: 'rgba(255, 255, 255, 0.1)', borderRadius: '8px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                <h3 style={{ margin: 0 }}>🎯 3D Preview</h3>
                {modelData && (
                  <button onClick={downloadSTL} style={{
                    background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', border: 'none',
                    color: 'white', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.8rem'
                  }}>📥 Download STL</button>
                )}
              </div>
              {modelData ? <ModelPreview modelData={modelData} /> : (
                <div style={{
                  height: '250px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  border: '2px dashed rgba(255, 255, 255, 0.2)', color: 'rgba(255, 255, 255, 0.6)'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🎯</div>
                    <div>Your 3D model will appear here</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Drawing Results */}
        {Object.keys(generatedDrawings).length > 0 && activeTab === 'drawings' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3>📐 Generated Drawings</h3>
              <button onClick={downloadDrawings} style={{
                background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', border: 'none',
                color: 'white', padding: '10px 20px', borderRadius: '8px', cursor: 'pointer', fontWeight: '600'
              }}>📥 Download All</button>
            </div>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px'
            }}>
              {Object.entries(generatedDrawings).map(([viewName, drawingData]) => (
                <div key={viewName} style={{
                  background: 'rgba(255, 255, 255, 0.1)', borderRadius: '8px', overflow: 'hidden'
                }}>
                  <div style={{
                    background: 'rgba(255, 255, 255, 0.1)', padding: '15px',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                  }}>
                    <h4 style={{ margin: 0 }}>
                      {drawingViews.find(v => v.id === viewName)?.icon} {drawingViews.find(v => v.id === viewName)?.name}
                    </h4>
                  </div>
                  <div style={{ padding: '20px', background: '#fff', minHeight: '200px' }}>
                    <div dangerouslySetInnerHTML={{ __html: atob(drawingData) }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
