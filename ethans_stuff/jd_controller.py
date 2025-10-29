# jd_speech_only.py  -- run:  python jd_speech_only.py
import socket, time

HOST, PORT = "127.0.0.1", 5000      # ARC TCP Script Server Raw (EZScript, Started)

def connect():
    while True:
        try:
            s = socket.create_connection((HOST, PORT), timeout=3)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("[ARC] connected")
            return s
        except OSError as e:
            print("[ARC] connect failed:", e, "retrying...")
            time.sleep(1.5)

class ARC:
    def __init__(self):
        self.s = connect()
        self.last = None

    def _send(self, ez: str):
        data = (ez + "\r\n").encode("utf-8")
        try:
            self.s.sendall(data)
            return self.s.recv(4096).decode("utf-8", errors="ignore").strip()
        except OSError:
            try: self.s.close()
            except: pass
            self.s = connect()
            self.s.sendall(data)
            return self.s.recv(4096).decode("utf-8", errors="ignore").strip()

    def say(self, text: str):
        safe = text.replace('"', "'")
        self._send(f'SayEZB("{safe}")')

    def set_label(self, label: str):
        safe = label.lower()
        self._send(f'$EmotionLabel = "{safe}"')

    def react(self, label: str):
        """Speak the label only when it changes."""
        lab = (label or "neutral").lower().strip()
        if lab == self.last:
            return
        self.last = lab

        # Speak the word
        self.set_label(lab)
        self._send(f'print("Emotion (new): {lab}")')
        self.say(lab.capitalize())

def manual_test():
    arc = ARC()
    print("Type: happy, sad, mad, surprised, neutral, or q to quit")
    while True:
        emo = input("emotion> ").strip().lower()
        if emo in ("q","quit","exit"): break
        if emo in ("happy","sad","mad","surprised","neutral"):
            arc.react(emo)
        else:
            print("valid: happy | sad | mad | surprised | neutral")

if __name__ == "__main__":
    manual_test()
