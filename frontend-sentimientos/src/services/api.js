import axios from "axios";

export const analizarSentimiento = async (texto) => {
    const response = await axios.post("https://proyecto-inteligencia-fwgk9e9y6.vercel.app/predict", {
        text: texto
    });
    return response.data;
};
