import { useState } from "react";
import Lottie from "lottie-react";

import { analizarSentimiento } from "./services/api";

// Animaciones
import loadingAnimation from "./lotties/loading.json";
import positiveAnimation from "./lotties/positive.json";
import negativeAnimation from "./lotties/negative.json";

function App() {
  const [texto, setTexto] = useState("");
  const [resultado, setResultado] = useState(null);
  const [loading, setLoading] = useState(false);

  const enviarTexto = async () => {
    if (!texto.trim()) return;

    setLoading(true);
    try {
      const data = await analizarSentimiento(texto);
      setResultado(data);
    } catch (error) {
      alert("Error al analizar el sentimiento 😥");
    } finally {
      setLoading(false);
    }
  };

  const getAnimation = () => {
    if (loading) return loadingAnimation;
    if (!resultado) return null;
    return resultado.sentimiento === "positive" ? positiveAnimation : negativeAnimation;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br h-full justify-center from-purple-600 to-indigo-500 flex flex-col items-center p-6 text-white">
      <div className="grid grid-cols-3 h-full w-full gap-6">
        <div className="flex-shrink-0 mr-4">
          <Lottie animationData={loadingAnimation} loop={true} style={{ height: 480, width: 480 }} />
        </div>
        <div>

          <h1 className="text-3xl font-bold mb-8 text-center drop-shadow-lg">
            Analizador de Sentimientos
          </h1>

          {/* Input */}
          <div className="w-full flex flex-col gap-4 mb-6">
            <textarea
              className="w-full h-80 p-4 rounded-xl text-black focus:outline-none focus:ring-2 focus:ring-cyan-400"
              rows={300}
              placeholder="Escribe lo que sientes..."
              value={texto}
              onChange={(e) => setTexto(e.target.value)}
            />
            <button
              onClick={enviarTexto}
              className="py-3 bg-cyan-400 rounded-full text-black font-semibold hover:bg-cyan-500 transition-all"
            >
              {loading ? "Analizando..." : "Analizar"}
            </button>
          </div>
        </div>

        {/* Chat / Resultado */}
        {(loading || resultado) && (
          <div className="w-full max-w-lg grid bg-white/20 backdrop-blur-md rounded-2xl p-4 shadow-lg">
            <div className="h-fit mx-auto">
              <Lottie className="mx-auto my-auto"  animationData={getAnimation()} loop={true} style={{ height: 180, width: 180 }} />
            </div>
            <div className="flex-1 text-left">
              {loading ? (
                <p className="text-white font-medium">Analizando tu mensaje...</p>
              ) : (
                <>
                  <h3 className={`text-xl font-bold ${resultado.sentimiento === "positive" ? "text-cyan-300" : "text-red-400"}`}>
                    {resultado.sentimiento.toUpperCase()}
                  </h3>
                  <p className="mt-2">{resultado.mensaje_gemini}</p>
                  <span className="text-sm text-white/70 mt-1 block">
                    Confianza: {resultado.confianza}
                  </span>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
