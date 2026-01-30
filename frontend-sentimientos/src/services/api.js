import axios from "axios";

export const analizarSentimiento = async (texto) => {
    const response = await axios.post("http://localhost:8000/predict", {
        text: texto
    });
    return response.data;
};
