import React, { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [context, setContext] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAsk = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setAnswer("");
    setContext("");
    try {
      const res = await fetch(
        `http://localhost:5000/ask?question=${encodeURIComponent(question)}`
      );
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Server error");
      }
      const data = await res.json();
      setAnswer(data.answer);
      setContext(data.retrievedContext);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="rag-container">
      <h1 className="rag-title">Ask your Question here</h1>
      <form className="rag-form" onSubmit={handleAsk}>
        <input
          className="rag-input"
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
          required
        />
        <button className="rag-button" type="submit" disabled={loading}>
          {loading ? "Asking..." : "Ask"}
        </button>
      </form>
      {error && <div className="rag-error">Error: {error}</div>}
      {answer && (
        <div className="rag-result">
          <h2>Here is your Answer</h2>
          <div className="rag-answer">{answer}</div>
          {context && (
            <>
              <h3>Retrieved Context</h3>
              <div className="rag-context">{context}</div>
            </>
          )}
        </div>
      )}
      <style>{`
        .rag-container {
          max-width: 600px;
          margin: 40px auto;
          padding: 42px 24px;
          background: #fff;
          border-radius: 12px;
          box-shadow: 0 2px 16px rgba(233, 32, 32, 0.08);
          font-family: 'Segoe UI', Arial, sans-serif;
        }
        .rag-title {
          text-align: center;
          margin-bottom: 24px;
          color: #2d3a4a;
        }
        .rag-form {
          display: flex;
          gap: 12px;
          margin-bottom: 24px;
        }
        .rag-input {
          flex: 1;
          padding: 10px 14px;
          border: 1px solid #cfd8dc;
          border-radius: 6px;
          font-size: 1rem;
        }
        .rag-button {
          padding: 10px 22px;
          background:rgb(58, 125, 172);
          color: #fff;
          border: none;
          border-radius: 6px;
          font-size: 1rem;
          cursor: pointer;
          transition: background 0.2s;
        }
        .rag-button:disabled {
          background: #90caf9;
          cursor: not-allowed;
        }
        .rag-error {
          color:rgb(9, 8, 8);
          margin-bottom: 18px;
          text-align: center;
        }
        .rag-result {
          margin-top: 24px;
          background: #f5f7fa;
          border-radius: 8px;
          padding: 18px 16px;
        }
        .rag-answer {
          font-size: 1.1rem;
          margin-bottom: 14px;
          color: #222;
        }
        .rag-context {
          background:rgb(20, 117, 182);
          padding: 12px;
          border-radius: 6px;
          font-size: 0.97rem;
          color: #444;
          white-space: pre-wrap;
        }
      `}</style>
    </div>
  );
}

export default App;