import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from brain import QLearningAgent

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
    log_msg = (f"DECISÃO: Solo {req.moisture}% às {req.hour}h | "
               f"Clima: {req.temp}°C, {req.air_humidity}% ur | "
               f"Ação: {idx} (fator {factor}) | Vol Final: {v_final}L")
    logger(log_msg)
    
    return {"action_idx": int(idx), "volume_final": v_final}

@app.post("/learn")
async def learn(req: LearnData):
    state_before = agent.get_state(req.moisture_before, req.hour_before, req.temp, req.air_humidity)
    state_after = agent.get_state(req.moisture_after, req.hour_after, req.temp, req.air_humidity)

    # Cálculo da Recompensa
    erro = abs(req.target_raw - req.moisture_after)
    reward = 10 - (erro * 0.5)
    if req.moisture_after > (req.target_raw + 10):
        reward -= 5

    agent.learn(state_before, req.action_idx, reward, state_after)

    # REGISTRO DE APRENDIZADO
    log_msg = (f"APRENDIZADO: Ação {req.action_idx} resultou em {req.moisture_after}% "
               f"(Alvo: {req.target_raw}%) | Recompensa: {reward:.2f}")
    logger(log_msg)
    
    return {"status": "learned", "reward": reward}