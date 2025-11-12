#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formato YAML esperado:
---------------------------------------
mt:
  states: [q0, q1, qf]
  input_alphabet: [a, b]
  tape_alphabet: [a, b, B, X]
  initial_state: q0
  accept_states: [qf]
  transitions:
    - state: q0
      read: [a, B]         # puede ser un símbolo o una lista de símbolos
      write: [a, B]        # si es lista, debe tener la misma longitud que 'read'
      move: R              # 'L' (izquierda), 'R' (derecha), 'N' (no mover)
      next: q1
  inputs:
    - "aabb"
    - "ab"

Notas :
- Permitimos que 'read' y 'write' sean un string (caso simple) o listas paralelas.
  Si son listas, expandimos en varias transiciones (read[i] -> write[i]).
- La máquina es de UNA cinta. El blanco por defecto es 'B'.
---------------------------------
Salida:
- Imprime por cada input todas las "descripciones instantáneas" (IDs) si --show-ids.
- Al finalizar, indica si ACEPTA o RECHAZA.
- Imprime la cinta final.(sin los blancos a los lados)
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# Intentamos importar PyYAML. Si no está, avisamos al usuario.
try:
    import yaml  # PyYAML
except Exception as e:
    print("ERROR: No se pudo importar 'yaml'. Instala PyYAML con:", file=sys.stderr)
    print("       pip install pyyaml", file=sys.stderr)
    raise

BLANK = 'B'  # símbolo de blanco por convenio

# ---------------------------------------------------------------------------
# Representación de la Cinta
# ---------------------------------------------------------------------------
class Tape:
    """
    Cinta infinita hacia ambos lados, representada como un diccionario esparso.
    - Solo almacenamos celdas que no son blancas para ahorrar memoria.
    - Las celdas no presentes se asumen como 'B' (blanco).
    - 'head' marca la posición actual del cabezal.
    """
    def __init__(self, input_str: str):
        self.cells: Dict[int, str] = {}
        for i, ch in enumerate(input_str):
            if ch != BLANK:
                self.cells[i] = ch
        self.head: int = 0

    def read(self) -> str:
        """Lee el símbolo bajo el cabezal (o 'B' si no hay nada escrito)."""
        return self.cells.get(self.head, BLANK)

    def write(self, symbol: str) -> None:
        """Escribe un símbolo en la posición del cabezal. Limpia si es 'B'."""
        if symbol == BLANK:
            self.cells.pop(self.head, None)  # escribir blanco equivale a borrar
        else:
            self.cells[self.head] = symbol

    def move_left(self) -> None:
        self.head -= 1

    def move_right(self) -> None:
        self.head += 1

    def snapshot(self, state: str, window: int = 20) -> str:
        """
        Devuelve una 'descripción instantánea' amigable:
        - una ventana de la cinta alrededor del cabezal,
        - con el estado actual y un caret (^) bajo el cabezal.
        window: cantidad de celdas hacia cada lado a mostrar.
        """
        # Determinamos rango a mostrar
        min_idx = min(self.cells.keys(), default=0)
        max_idx = max(self.cells.keys(), default=0)
        # Ampliamos para siempre mostrar el cabezal
        min_idx = min(min_idx, self.head - window)
        max_idx = max(max_idx, self.head + window)

        tape_str = []
        head_line = []

        for i in range(min_idx, max_idx + 1):
            sym = self.cells.get(i, BLANK)
            # Representamos el estado "incrustado" sobre el símbolo actual
            if i == self.head:
                tape_str.append(f"[{state}:{sym}]")
                head_line.append("^".center(len(f"[{state}:{sym}]")))
            else:
                tape_str.append(sym)
                head_line.append(" ".center(len(sym)))

        return "".join(tape_str) + "\n" + "".join(head_line)

# ---------------------------------------------------------------------------
# Máquina de Turing: estados y transiciones
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Transition:
    """Una transición simple: (estado, read) -> (write, move, next_state)."""
    state: str
    read: str
    write: str
    move: str   # 'L' | 'R' | 'N'
    next: str

class TuringMachine:
    """
    Implementación directa de MT de una cinta con:
    - estados finitos,
    - alfabeto de entrada/tape,
    - estado inicial y de aceptación,
    - transiciones deterministas (un símbolo leído -> una acción).
    """
    def __init__(
        self,
        states: List[str],
        input_alphabet: List[str],
        tape_alphabet: List[str],
        initial_state: str,
        accept_states: List[str],
        transitions: List[Transition],
    ):
        self.states = set(states)
        self.input_alphabet = set(input_alphabet)
        self.tape_alphabet = set(tape_alphabet)
        self.initial_state = initial_state
        self.accept_states = set(accept_states)

        # Tabla de transiciones: (state, read) -> Transition
        table: Dict[Tuple[str, str], Transition] = {}
        for t in transitions:
            if t.move not in ('L', 'R', 'N'):
                raise ValueError(f"Movimiento inválido: {t.move}")
            # opcional: validaciones de símbolos/estados
            key = (t.state, t.read)
            if key in table:
                raise ValueError(f"Transición duplicada para {key}")
            table[key] = t
        self.delta = table

    def step(self, state: str, tape: Tape) -> Tuple[str, bool]:
        """
        Ejecuta UN paso de la máquina:
        - Lee el símbolo bajo el cabezal,
        - Busca transición (state, symbol),
        - Si no hay transición: HALT (rechazo si no es estado de aceptación),
        - Si hay: escribe, mueve, cambia estado.
        Devuelve (nuevo_estado, halted).
        halted = True cuando no hay transición o cuando entramos a un estado de aceptación.
        """
        # Si ya estamos en estado de aceptación, podemos haltear (aceptar)
        if state in self.accept_states:
            return (state, True)

        symbol = tape.read()
        key = (state, symbol)
        t = self.delta.get(key)
        if t is None:
            # No hay transición: la máquina se detiene (rechazo si no es aceptador)
            return (state, True)

        # Aplicamos transición
        tape.write(t.write)
        if t.move == 'L':
            tape.move_left()
        elif t.move == 'R':
            tape.move_right()
        # N = no mover

        return (t.next, False)

    def run(self, input_str: str, max_steps: int = 10000, show_ids: bool = True) -> Tuple[bool, int]:
        """
        Ejecuta la MT sobre una cadena de entrada.
        - Devuelve (accepted, steps).
        - 'accepted' es True si la máquina detiene en estado de aceptación.
        - 'steps' indica cuántos pasos ejecutó.
        - Si show_ids es True, imprime las descripciones instantáneas.
        """
        tape = Tape(input_str)
        state = self.initial_state

        if show_ids:
            print("== INICIO ==")
            print(tape.snapshot(state))

        steps = 0
        while steps < max_steps:
            state, halted = self.step(state, tape)
            steps += 1
            if show_ids:
                print(f"-- paso {steps} --")
                print(tape.snapshot(state))
            if halted:
                break

        accepted = (state in self.accept_states)
        if steps >= max_steps and not accepted:
            print(f"[Aviso] Se alcanzó el máximo de pasos ({max_steps}) sin aceptar.")
        final_tape = ''.join(tape.cells.get(i, 'B') for i in range(min(tape.cells.keys(), default=0),max(tape.cells.keys(), default=0) + 1))
        print(f"Cinta final: {final_tape}")
        return accepted, steps

# ---------------------------------------------------------------------------
# Carga desde YAML y expansión de transiciones simples
# ---------------------------------------------------------------------------
def load_tm_from_yaml(path: str) -> Tuple[TuringMachine, List[str]]:
    """
    Lee un archivo YAML y devuelve (Máquina de Turing, lista_de_inputs).
    También valida y 'expande' transiciones donde read/write sean listas paralelas.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "mt" not in data:
        raise ValueError("YAML inválido: falta llave 'mt' en la raíz.")

    cfg = data["mt"]

    states = cfg.get("states", [])
    input_alphabet = cfg.get("input_alphabet", [])
    tape_alphabet = cfg.get("tape_alphabet", [])
    initial_state = cfg.get("initial_state")
    accept_states = cfg.get("accept_states", [])
    raw_transitions = cfg.get("transitions", [])
    inputs = cfg.get("inputs", [])

    if initial_state is None:
        raise ValueError("YAML inválido: falta 'initial_state'.")

    transitions: List[Transition] = []

    for idx, t in enumerate(raw_transitions):
        state = t["state"]
        move = t["move"]
        next_state = t["next"]
        read = t["read"]
        write = t["write"]

        # Caso 1: read y write son strings simples
        if isinstance(read, str) and isinstance(write, str):
            transitions.append(Transition(state=state, read=read, write=write, move=move, next=next_state))
            continue

        # Caso 2: read y write son listas paralelas: expandimos elemento a elemento
        if isinstance(read, list) and isinstance(write, list):
            if len(read) != len(write):
                raise ValueError(f"Transición {idx}: 'read' y 'write' deben ser listas de igual longitud.")
            for r_sym, w_sym in zip(read, write):
                transitions.append(Transition(state=state, read=r_sym, write=w_sym, move=move, next=next_state))
            continue

        # Cualquier otra combinación la consideramos inválida para mantenerlo simple
        raise ValueError(
            f"Transición {idx}: formato no soportado. Usa 'read' y 'write' como string o listas paralelas."
        )

    tm = TuringMachine(
        states=states,
        input_alphabet=input_alphabet,
        tape_alphabet=tape_alphabet,
        initial_state=initial_state,
        accept_states=accept_states,
        transitions=transitions,
    )
    return tm, inputs

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    yaml_path = r"C:\Users\Joabh\Documents\GitHub\maquinaturing_joabJorge\mt_alteradora_swap.yml"
    max_steps = 10000
    show_ids = True


    tm, inputs = load_tm_from_yaml(yaml_path)

    for s in inputs:
        print("=" * 60)
        print(f"Input: {repr(s)}")
        accepted, steps = tm.run(s, max_steps=max_steps, show_ids=show_ids)
        print(f"Resultado: {'ACEPTA' if accepted else 'RECHAZA'} en {steps} paso(s).")
    print("=" * 60)

if __name__ == "__main__":
    main()
