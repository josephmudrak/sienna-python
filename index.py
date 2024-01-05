import asyncio
import base64
import json
import numpy
import os
import pygame
import shutil
import sounddevice
import subprocess
import websockets

from deepgram import (
	DeepgramClient,
	LiveTranscriptionEvents,
	LiveOptions,
	Microphone
)

from dotenv import load_dotenv
from io import BytesIO
from openai import AsyncOpenAI
from pydub import AudioSegment
from pydub.playback import play

load_dotenv()

# Define API keys and voice ID
ELEVENLABS_API_KEY	= os.environ.get("ELEVENLABS_API_KEY")
OPENAI_API_KEY		= os.environ.get("OPENAI_API_KEY")
VOICE_ID			= "oWAxZDx7w5VEj9dCyTzz"

# Set OpenAI API key
aclient	= AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialise pygame mixer
pygame.mixer.init()

def is_installed(lib_name):
	return shutil.which(lib_name) is not None

# Split text into chunks, ensuring to not break sentences
async def text_chunker(chunks):
	splitters	= (
		".", ",", "?", "!", ";", ":", "—",
		"-", "(", ")", "[", "]", "}", " "
		)
	buffer		= ""

	async for text in chunks:
		if buffer.endswith(splitters):
			yield buffer + " "
			buffer	= text
		elif text.startswith(splitters):
			yield buffer + text[0] + " "
			buffer	= text[1:]
		else:
			buffer	+= text

	if buffer:
		yield buffer + " "

# Stream audio data
async def stream(audio_stream):
	print("Started streaming audio")

	# Collect audio data from async generator
	audio_data	= b''

	async for chunk in audio_stream:
		audio_data += chunk.read()

	try:
		# Create AudioSegment from collected data
		audio_segment	= AudioSegment.from_file(
			BytesIO(audio_data),
			format="mp3"
		)

		# Play audio
		play(audio_segment)
	except Exception as e:
		print(f"Error during playback: {e}")

# Send text to ElevenLabs API & stream returned audio
async def text_to_speech_input_streaming(voice_id, text_iterator):
	uri	= f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1"

	async with websockets.connect(uri) as websocket:
		await websocket.send(json.dumps({
			"text":				" ",
			"voice_settings":	{"stability": 0.5, "similarity_boost": 0.8},
			"xi_api_key":		ELEVENLABS_API_KEY
		}))

		# Listen to websocket for audio data & stream
		async def listen():
			while True:
				try:
					message	= await websocket.recv()
					data	= json.loads(message)

					if data.get("audio"):
						# Convert audio to streamable raw data
						yield BytesIO(base64.b64decode(data["audio"]))
					elif data.get("isFinal"):
						break

				except websockets.exceptions.ConnectionClosed:
					print("Connection closed")
					break

		listen_task	= asyncio.create_task(stream(listen()))

		async for text in text_chunker(text_iterator):
			await websocket.send(json.dumps({
				"text":						text,
				"try_trigger_generation":	True
			}))

		await websocket.send(json.dumps({"text": ""}))

		await listen_task

# Retrieve text from OpenAI & pass to TTS function
async def chat_completion(query):
	response	= await aclient.chat.completions.create(
		model="gpt-4-1106-preview",
		messages=[{"role": "user", "content": query}],
		temperature=1,
		stream=True,
		max_tokens=15
	)

	async def text_iterator():
		async for chunk in response:
			delta	= chunk.choices[0].delta
			if delta.content:
				yield delta.content

	await text_to_speech_input_streaming(VOICE_ID, text_iterator())

async def main():
	try:
		# Create Deepgram client
		deepgram		= DeepgramClient()
		dg_connection	= deepgram.listen.live.v("1")

		def on_message(self, result, **kwargs):
			sentence	= result.channel.alternatives[0].transcript
			if len(sentence) == 0:
				return
			print(f"transcription: {sentence}")
			asyncio.run(chat_completion(sentence))

		def on_error(self, error, **kwargs):
			print(f"\n\n{error}\n\n")

		dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
		dg_connection.on(LiveTranscriptionEvents.Error, on_error)

		options	= LiveOptions(
			model="nova-2",
			smart_format=True,
			language="en-US",
			encoding="linear16",
			channels=1,
			sample_rate=16000
		)

		dg_connection.start(options)

		microphone	= Microphone(dg_connection.send)

		# Start microphone
		microphone.start()

		# Wait until finished
		input("Press Enter to stop recording...\n\n")

		# wait for microphone to close
		microphone.finish()

		# Indicate end
		dg_connection.finish()

		print("Finished")

	except Exception as e:
		print(f"Could not open socket: {e}")
		return

# Main execution
if __name__ == "__main__":
	asyncio.run(main())
