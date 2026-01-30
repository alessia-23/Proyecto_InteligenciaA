import axios from "axios";

// Accedemos a la variable de entorno
const API_URL = import.meta.env.VITE_BACKEND_URL;

export const analizarSentimiento = async (texto) => {
    try {
        const response = await axios.post(`${API_URL}/predict`, {
            text: texto
        });
        return response.data;
    } catch (error) {
        console.error("Error al analizar sentimiento:", error);
        throw error;
    }
};