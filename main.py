import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from brain import QLearningAgent
import os
import numpy as np
from fastapi import HTTPException

app = FastAPI()
agent = QLearningAgent()

# Nome do arquivo de log
LOG_FILE = "ia_history.txt"

def logger(message: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# Modelos Pydantic (mantidos conforme a última versão)
class DecideData(BaseModel):
    moisture: float
    hour: int
    temp: float
    air_humidity: float
    volume_ab: float

class LearnData(BaseModel):
    moisture_before: float
    hour_before: int
    temp: float
    air_humidity: float
    action_idx: int
    volume_applied: float
    moisture_after: float
    hour_after: int
    target_raw: float

@app.post("/decide")
async def decide(req: DecideData):
    idx, factor = agent.decide(req.moisture, req.hour, req.temp, req.air_humidity)
    v_final = round(req.volume_ab * factor, 3)
    
    # REGISTRO DE DECISÃO
    log_msg = (f"DECISÃO: Solo {req.moisture}L às {req.hour}h, | "
               f"Clima: {req.temp}°C, {req.air_humidity}% ur | "
               f"Ação: {idx} (fator {factor}) | Vol Final: {v_final}L")
    logger(log_msg)
    
    return {"action_idx": int(idx), "volume_final": v_final}

@app.post("/learn")
async def learn(req: LearnData):
    print(req)
    state_before = agent.get_state(req.moisture_before, req.hour_before, req.temp, req.air_humidity)
    state_after = agent.get_state(req.moisture_after, req.hour_after, req.temp, req.air_humidity)

    # Cálculo da Recompensa
    erro = abs(req.target_raw - req.moisture_after)
    reward = 10 - (erro * 0.5)
    if req.moisture_after > (req.target_raw + 10):
        reward -= 5
    print(f"Recompensa calculada: {reward:.2f} (Erro: {erro:.2f}%)")
    agent.learn(state_before, req.action_idx, reward, state_after)

    # REGISTRO DE APRENDIZADO
    log_msg = (f"APRENDIZADO: Ação {req.action_idx}, add {req.volume_applied}L resultou em {req.moisture_after}L "
               f"(Alvo: {req.target_raw}L) | Recompensa: {reward:.2f}")
    logger(log_msg)
    
    return {"status": "learned", "reward": reward}

@app.delete("/reset-agent")
async def reset_agent():
    try:
        # Em vez de agent.q_table = {}, use o método do agente
        agent.reset_memory() 
        
        log_msg = "SISTEMA: Memória da IA resetada."
        logger(log_msg)
        return {"status": "success", "message": log_msg}

    except Exception as e:
        logger(f"ERRO ao resetar agente: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/dashboard")
async def get_ia_dashboard():
    try:
        qt = agent.q_table
        total_cells = qt.size
        
        # 1. Filtro de células exploradas (onde o valor não é zero)
        learned_mask = qt != 0
        learned_cells = np.count_nonzero(learned_mask)
        coverage = (learned_cells / total_cells) * 100

        # 2. Preferência de Ações (Quais multiplicadores a IA mais valoriza)
        # Fazemos a média dos valores Q para cada uma das 5 ações
        avg_q_per_action = np.mean(qt, axis=(0, 1, 2, 3))
        action_report = {
            f"Ação {i} (x{agent.actions[i]})": round(float(avg_q_per_action[i]), 6)
            for i in range(len(agent.actions))
        }

        # 3. Distribuição de Conhecimento por Turno (Manhã vs Noite)
        # q_shape: (Umidade, Hora, Temp, Ar, Ações) -> Hora está no índice 1
        knowledge_per_hour = np.sum(np.abs(qt), axis=(0, 2, 3, 4))
        morning_knowledge = float(np.sum(knowledge_per_hour[6:12])) # 6h às 11h
        afternoon_knowledge = float(np.sum(knowledge_per_hour[12:18])) # 12h às 17h
        night_knowledge = float(np.sum(knowledge_per_hour[18:24]) + np.sum(knowledge_per_hour[0:6]))

        # 4. Valores Extremos (Para detectar se a recompensa está explodindo)
        max_q = float(np.max(qt))
        min_q = float(np.min(qt))

        return {
            "status_geral": {
                "cobertura_da_memoria": f"{coverage:.4f}%",
                "total_conexoes_aprendidas": int(learned_cells),
                "volume_total_q": float(np.sum(np.abs(qt)))
            },
            "preferencia_de_algoritmo": action_report,
            "foco_do_aprendizado": {
                "manha": round(morning_knowledge, 2),
                "tarde": round(afternoon_knowledge, 2),
                "noite": round(night_knowledge, 2),
                "turno_mais_treinado": "Manhã" if morning_knowledge > afternoon_knowledge else "Tarde/Noite"
            },
            "estabilidade": {
                "maior_recompensa_acumulada": round(max_q, 4),
                "menor_recompensa_acumulada": round(min_q, 4),
                "media_q_global": float(np.mean(qt[learned_mask])) if learned_cells > 0 else 0
            }
        }
    except Exception as e:
        logger(f"ERRO ao gerar dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))