"""
UMI Serial axis[4] reader. This script reads only axis[4] data from a serial port.
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import re
import time
import serial

from serial import SerialException
from typing import Optional

class Esp32gripperReader:
	def __init__(
		self,
		port: str = "/dev/ttyACM0",
		baudrate: int = 115200,
		read_hz: float = 200.0,
		axis_center: float = 128.0,
		axis_scale: float = 127.0,
	) -> None:
		self._port = port
		self._baudrate = baudrate
		self._read_hz = read_hz
		self._axis_center = axis_center
		self._axis_scale = axis_scale
		self._logger = logging.getLogger(self.__class__.__name__)
		self.axis_4: Optional[float] = None
		self._last_logged_axis_4 = self.axis_4
		self._last_read_time = 0.0
		self._serial: Optional[serial.Serial] = None
		self.connect()

		self._logger.info(
			"UART axis[4] reader started: port=%s baud=%d"
			% (self._port, self._baudrate)
		)

	def _open_serial(self) -> Optional[serial.Serial]:
		try:
			return serial.Serial(self._port, self._baudrate, timeout=0.1)
		except SerialException as exc:
			self._logger.error("Failed to open serial %s: %s" % (self._port, str(exc)))
			return None

	@property
	def is_connected(self) -> bool:
		return self._serial is not None and self._serial.is_open

	def connect(self) -> bool:
		if self.is_connected:
			return True
		self._serial = self._open_serial()
		if self._serial is not None:
			try:
				# Drop stale bytes so first reads reflect current gripper state.
				self._serial.reset_input_buffer()
			except SerialException:
				pass
		return self.is_connected

	def read_once(self) -> Optional[float]:
		if self._serial is None:
			self.connect()
			return self.axis_4

		latest_axis_4 = self._read_latest_axis_from_buffer()
		if latest_axis_4 is None or latest_axis_4 == self.axis_4:
			return self.axis_4

		self.axis_4 = latest_axis_4
		self._last_read_time = time.time()
		if self.axis_4 != self._last_logged_axis_4:
			self._logger.info("axis[4]=%.6f" % self.axis_4)
			self._last_logged_axis_4 = self.axis_4
		return self.axis_4

	def _read_latest_axis_from_buffer(self) -> Optional[float]:
		if self._serial is None:
			return None

		latest_axis_4: Optional[float] = None
		try:
			raw_line = self._serial.readline()
			while raw_line:
				parsed_axis_4 = self._parse_axis_4_from_line(raw_line)
				if parsed_axis_4 is not None:
					latest_axis_4 = parsed_axis_4
				if self._serial.in_waiting <= 0:
					break
				raw_line = self._serial.readline()
		except SerialException as exc:
			self._logger.error("Serial read error: %s" % str(exc))
			self.disconnect()
			return None
		return latest_axis_4

	def _parse_axis_4_from_line(self, raw_line: bytes) -> Optional[float]:
		line = raw_line.decode("utf-8", errors="ignore").strip()
		if not line:
			return None
		raw_axis_4 = 255 if "[IDLE]" in line else self._extract_axis_4(line)
		if raw_axis_4 is None:
			return None
		return self._normalize_axis(raw_axis_4)
		
	def _extract_axis_4(self, line: str) -> Optional[int]:
		match = re.search(r"\brx\s*=\s*(-?\d+)", line, flags=re.IGNORECASE)
		if match is None:
			return None
		return int(match.group(1))

	def _normalize_axis(self, raw_value: int) -> float:
		if self._axis_scale <= 0.0:
			return float(raw_value)
		normalized = (float(raw_value) - self._axis_center) / self._axis_scale
		return max(-1.0, min(1.0, normalized))

	def close(self) -> None:
		if self._serial is None:
			return
		try:
			self._serial.close()
		except SerialException:
			pass
		self._serial = None

	def disconnect(self) -> None:
		self.close()

	def get_devices(self) -> list[str]:
		if not self.is_connected:
			return []
		return [self._port]

	def get_device_info(self, device_name: Optional[str] = None) -> dict:
		return {
			"port": self._port,
			"baudrate": self._baudrate,
			"read_hz": self._read_hz,
			"axis_center": self._axis_center,
			"axis_scale": self._axis_scale,
			"is_connected": self.is_connected,
			"last_read_time": self._last_read_time,
			"axis_4": self.axis_4,
		}
