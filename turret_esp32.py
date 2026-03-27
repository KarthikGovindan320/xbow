import network, ujson, time, _thread, socket
from machine import Pin, PWM, SoftI2C, ADC

WIFI_SSID            = "YOUR_WIFI_SSID"
WIFI_PASSWORD        = "YOUR_WIFI_PASSWORD"
DRONE_WIFI_SSID      = "YOUR_DRONE_HOTSPOT_SSID"

UDP_PORT             = 5005
STALE_SECS           = 2.0

YAW_PIN              = 18
PITCH_PIN            = 19
TRACK_LASER_PIN      = 22
BULLET_LASER_PIN     = 23
FIRE_BUTTON_PIN      = 21
LCD_SDA_PIN          = 26
LCD_SCL_PIN          = 27
BUZZER_PIN           = 25
LCD_I2C_ADDR         = 0x27

MODE_BUTTON_PIN      = 13
JOYSTICK_PAN_PIN     = 34
JOYSTICK_TILT_PIN    = 35
JOY_DEADBAND         = 150
JOY_MAX_SPEED        = 1.0
JOY_ADC_SMOOTH       = 0.4

ULTRASONIC_TRIG_PIN  = 32
ULTRASONIC_ECHO_PIN  = 33

SERVO_MIN_US         = 500
SERVO_MAX_US         = 2500
PWM_FREQ             = 50
YAW_LIMIT_DEG        = 90
PITCH_LIMIT_DEG      = 45

SERVO_INTERVAL_MS    = 5
SMOOTH               = 0.35
DEADBAND_DEG         = 0.5
TIMEOUT_HOME         = STALE_SECS

FIRE_DURATION_S      = 2
DEBOUNCE_MS          = 200
LCD_INTERVAL_MS      = 120

AUTO_FIRE_ENABLED        = True
AUTO_FIRE_STABLE_FRAMES  = 60
AUTO_FIRE_MIN_M          = 0.3
AUTO_FIRE_MAX_M          = 5.0
AUTO_FIRE_COOLDOWN_S     = 5.0

BURST_COUNT          = 3
BURST_ON_MS          = 400
BURST_OFF_MS         = 200

shared = {
    "yaw":        [0.0],
    "pitch":      [0.0],
    "distance":   [0.0],
    "us_dist":    [0.0],
    "fresh":      [False],
    "locked":     [False],
    "net_status": ["INIT"],
    "manual":     [False],
    "auto_fire":  [AUTO_FIRE_ENABLED],
    "firing":     [False],
    "cease_fire": [False],
}


class Servo:
    def __init__(self, pin_num, freq=PWM_FREQ,
                 min_us=SERVO_MIN_US, max_us=SERVO_MAX_US,
                 limit_deg=90, name="SERVO"):
        self.pwm      = PWM(Pin(pin_num), freq=freq)
        self.min_us   = min_us
        self.max_us   = max_us
        self.limit    = limit_deg
        self.name     = name
        self._current = 0.0
        self._period  = 1_000_000 // freq
        self.write(0.0)

    def _deg_to_duty(self, deg):
        deg = max(-self.limit, min(self.limit, float(deg)))
        us  = self.min_us + (self.max_us - self.min_us) * ((deg + 90.0) / 180.0)
        return int(us / self._period * 1023)

    def write(self, deg):
        deg = max(-self.limit, min(self.limit, float(deg)))
        self.pwm.duty(self._deg_to_duty(deg))
        self._current = deg

    def current(self):
        return self._current

    def deinit(self):
        self.pwm.deinit()


class LCD:
    BL = 0x08
    EN = 0x04
    RS = 0x01

    def __init__(self, i2c, addr=0x27, cols=16):
        self.i2c  = i2c
        self.addr = addr
        self.cols = cols
        self._bl  = self.BL
        self._init()

    def _w(self, b):
        self.i2c.writeto(self.addr, bytes([b | self._bl]))

    def _pulse(self, d):
        self._w(d | self.EN)
        time.sleep_us(2)
        self._w(d & ~self.EN)
        time.sleep_us(50)

    def _send4(self, d, mode):
        self._pulse((d & 0xF0) | mode)
        self._pulse(((d << 4) & 0xF0) | mode)

    def _cmd(self, c):
        self._send4(c, 0)
        if c < 4:
            time.sleep_ms(2)

    def _init(self):
        time.sleep_ms(45)
        for _ in range(3):
            self._pulse(0x30)
            time.sleep_ms(5)
        self._pulse(0x20)
        time.sleep_ms(1)
        self._cmd(0x28)
        self._cmd(0x0C)
        self._cmd(0x06)
        self._cmd(0x01)
        time.sleep_ms(2)

    def print_line(self, text, row=0):
        self._cmd(0x80 | ([0x00, 0x40][row % 2]))
        txt = str(text)[:self.cols]
        txt = txt + " " * (self.cols - len(txt))
        for ch in txt:
            self._send4(ord(ch), self.RS)


def lcd_thread(lcd):
    prev0    = prev1    = ""
    was_manual  = False
    was_firing  = False

    while True:
        manual = shared["manual"][0]
        firing = shared["firing"][0]
        cease  = shared["cease_fire"][0]

        if cease:
            r0 = "CEASE FIRE!!!!!!"
            r1 = "CIVILIAN/STICKER"
            if r0 != prev0:
                try: lcd.print_line(r0, 0)
                except: pass
                prev0 = r0
            if r1 != prev1:
                try: lcd.print_line(r1, 1)
                except: pass
                prev1 = r1
            time.sleep_ms(LCD_INTERVAL_MS)
            continue

        if firing and not was_firing:
            try:
                lcd.print_line("*** FIRING ***  ", 0)
                lcd.print_line("  BURST MODE    ", 1)
            except: pass
            prev0     = "*** FIRING ***  "
            prev1     = "  BURST MODE    "
            was_firing = True
            time.sleep_ms(LCD_INTERVAL_MS)
            continue
        if firing:
            time.sleep_ms(LCD_INTERVAL_MS)
            continue
        if was_firing:
            prev0 = prev1 = ""
            was_firing = False

        if manual:
            if not was_manual:
                try:
                    lcd.print_line("** MANUAL MODE**", 0)
                    lcd.print_line(" Joystick ctrl  ", 1)
                except: pass
                prev0     = "** MANUAL MODE**"
                prev1     = " Joystick ctrl  "
                was_manual = True
            time.sleep_ms(LCD_INTERVAL_MS)
            continue
        if was_manual:
            prev0 = prev1 = ""
            was_manual = False

        locked = shared["locked"][0]
        row0   = "TRACKING" if locked else "Idle"

        if locked:
            y   = shared["yaw"][0]
            d   = shared["us_dist"][0] if shared["us_dist"][0] > 0.0 else shared["distance"][0]
            ys  = ("+" if y >= 0 else "") + "{:.1f}".format(y)
            af  = "A" if shared["auto_fire"][0] else "M"
            row1 = af + " Y:" + ys + " D:" + "{:.1f}".format(d) + "m"
        else:
            row1 = shared["net_status"][0][:16]

        if row0 != prev0:
            try: lcd.print_line(row0, 0)
            except: pass
            prev0 = row0
        if row1 != prev1:
            try: lcd.print_line(row1, 1)
            except: pass
            prev1 = row1

        time.sleep_ms(LCD_INTERVAL_MS)


def init_ultrasonic():
    trig = Pin(ULTRASONIC_TRIG_PIN, Pin.OUT, value=0)
    echo = Pin(ULTRASONIC_ECHO_PIN, Pin.IN)
    return trig, echo


def read_ultrasonic(trig, echo, timeout_us=30000):
    trig.off()
    time.sleep_us(2)
    trig.on()
    time.sleep_us(10)
    trig.off()

    t0 = time.ticks_us()
    while echo.value() == 0:
        if time.ticks_diff(time.ticks_us(), t0) > timeout_us:
            return -1.0

    t1 = time.ticks_us()
    while echo.value() == 1:
        if time.ticks_diff(time.ticks_us(), t1) > timeout_us:
            return -1.0

    t2      = time.ticks_us()
    dur_us  = time.ticks_diff(t2, t1)
    dist_m  = (dur_us / 2.0) / 10000.0 * 0.343
    return dist_m


def ultrasonic_thread(trig, echo):
    while True:
        d = read_ultrasonic(trig, echo)
        if d >= 0.0:
            shared["us_dist"][0] = d
        time.sleep_ms(60)


def scan_for_drone_wifi(lcd=None):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    time.sleep(2)
    attempt = 0
    while True:
        attempt += 1
        if lcd:
            try:
                lcd.print_line("RF Scan...     ", 0)
                lcd.print_line("attempt " + str(attempt), 1)
            except: pass
        try:
            nets = wlan.scan()
            ssids = [
                (n[0].decode("utf-8") if isinstance(n[0], bytes) else n[0])
                for n in nets
            ]
            if DRONE_WIFI_SSID in ssids:
                if lcd:
                    try:
                        lcd.print_line("RF FOUND!      ", 0)
                        lcd.print_line("Arming...      ", 1)
                    except: pass
                time.sleep_ms(500)
                return wlan
        except Exception as e:
            pass
        time.sleep_ms(3000)


def connect_wifi(wlan=None):
    if wlan is None:
        wlan = network.WLAN(network.STA_IF)
    try: wlan.disconnect()
    except: pass
    wlan.active(False)
    time.sleep_ms(300)
    wlan.active(True)
    time.sleep(2)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    for i in range(20):
        if wlan.isconnected():
            return wlan
        time.sleep_ms(500)
    raise RuntimeError("WiFi connect failed")


def udp_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.setblocking(False)
    shared["net_status"][0] = "LISTEN"
    pkt_count = 0

    while True:
        try:
            data, addr = sock.recvfrom(256)
            obj = ujson.loads(data)
            shared["yaw"][0]      = float(obj["y"])
            shared["pitch"][0]    = float(obj["p"])
            shared["distance"][0] = float(obj.get("d", 0.0))

            cease = bool(obj.get("cease", 0))
            if cease and not shared["cease_fire"][0]:
                shared["cease_fire"][0] = True
            elif not cease and shared["cease_fire"][0]:
                shared["cease_fire"][0] = False

            shared["fresh"][0]    = True
            shared["net_status"][0] = "OK"
            pkt_count += 1
        except OSError:
            pass
        except Exception:
            shared["net_status"][0] = "ERR"
        time.sleep_ms(1)


def joy_to_delta(raw):
    offset = raw - 2048
    if abs(offset) < JOY_DEADBAND:
        return 0.0
    norm = (offset - (JOY_DEADBAND if offset > 0 else -JOY_DEADBAND)) / (2048 - JOY_DEADBAND)
    norm = max(-1.0, min(1.0, norm))
    return norm * JOY_MAX_SPEED


def do_burst_fire(track_laser, bullet_laser, buzzer,
                  servo_yaw, servo_pitch,
                  cur_yaw_ref, cur_pitch_ref,
                  label="[FIRE]"):
    if shared["cease_fire"][0]:
        return

    shared["firing"][0] = True
    track_laser.off()

    for burst_idx in range(BURST_COUNT):
        bullet_laser.on()
        burst_end = time.ticks_add(time.ticks_ms(), BURST_ON_MS)
        while time.ticks_diff(burst_end, time.ticks_ms()) > 0:
            if shared["cease_fire"][0]:
                bullet_laser.off()
                shared["firing"][0] = False
                track_laser.on()
                return
            if shared["fresh"][0]:
                shared["fresh"][0] = False
                ty = max(-YAW_LIMIT_DEG,   min(YAW_LIMIT_DEG,   shared["yaw"][0]))
                tp = max(-PITCH_LIMIT_DEG, min(PITCH_LIMIT_DEG, shared["pitch"][0]))
                ey = ty - cur_yaw_ref[0]
                ep = tp - cur_pitch_ref[0]
                ey = 0.0 if abs(ey) < DEADBAND_DEG else ey
                ep = 0.0 if abs(ep) < DEADBAND_DEG else ep
                if ey or ep:
                    cur_yaw_ref[0]   += SMOOTH * ey
                    cur_pitch_ref[0] += SMOOTH * ep
                    servo_yaw.write(cur_yaw_ref[0])
                    servo_pitch.write(cur_pitch_ref[0])
            time.sleep_ms(SERVO_INTERVAL_MS)

        bullet_laser.off()
        if burst_idx < BURST_COUNT - 1:
            time.sleep_ms(BURST_OFF_MS)

    buzzer.on()
    time.sleep_ms(500)
    buzzer.off()

    shared["firing"][0] = False
    track_laser.on()


def main():
    i2c   = SoftI2C(sda=Pin(LCD_SDA_PIN), scl=Pin(LCD_SCL_PIN), freq=400_000)
    found = i2c.scan()
    lcd   = None
    if LCD_I2C_ADDR in found:
        try:
            lcd = LCD(i2c, addr=LCD_I2C_ADDR)
            lcd.print_line(" TURRET SYSTEM ", 0)
            lcd.print_line("  Booting...   ", 1)
        except Exception:
            pass

    servo_yaw   = Servo(YAW_PIN,   limit_deg=YAW_LIMIT_DEG,   name="YAW")
    servo_pitch = Servo(PITCH_PIN, limit_deg=PITCH_LIMIT_DEG, name="PITCH")

    track_laser  = Pin(TRACK_LASER_PIN,  Pin.OUT, value=0)
    bullet_laser = Pin(BULLET_LASER_PIN, Pin.OUT, value=0)
    fire_btn     = Pin(FIRE_BUTTON_PIN,  Pin.IN,  Pin.PULL_UP)
    mode_btn     = Pin(MODE_BUTTON_PIN,  Pin.IN,  Pin.PULL_UP)
    buzzer       = Pin(BUZZER_PIN,       Pin.OUT, value=0)

    adc_pan  = ADC(Pin(JOYSTICK_PAN_PIN))
    adc_tilt = ADC(Pin(JOYSTICK_TILT_PIN))
    adc_pan.atten(ADC.ATTN_11DB)
    adc_tilt.atten(ADC.ATTN_11DB)

    us_trig, us_echo = init_ultrasonic()
    _thread.start_new_thread(ultrasonic_thread, (us_trig, us_echo))

    wlan = scan_for_drone_wifi(lcd=lcd)
    try:
        connect_wifi(wlan)
    except RuntimeError:
        if lcd:
            try: lcd.print_line("NO WIFI  HALT  ", 0)
            except: pass
        return

    _thread.start_new_thread(udp_thread, ())

    if lcd:
        _thread.start_new_thread(lcd_thread, (lcd,))

    cur_yaw_ref   = [0.0]
    cur_pitch_ref = [0.0]
    cur_yaw       = 0.0
    cur_pitch     = 0.0

    last_ts         = time.time()
    last_fire_btn   = 0
    last_mode_btn   = 0
    prev_mode_val   = 1
    locked          = False
    at_home         = True
    manual          = False
    lc              = 0
    smooth_pan      = 2048.0
    smooth_tilt     = 2048.0

    stable_frames        = 0
    last_auto_fire_time  = 0.0
    AUTO_STABLE_YAW_DEG   = 2.0
    AUTO_STABLE_PITCH_DEG = 2.0

    while True:
        t0     = time.ticks_ms()
        lc    += 1
        now_ms = time.ticks_ms()

        cur_yaw   = cur_yaw_ref[0]
        cur_pitch = cur_pitch_ref[0]

        mode_val = mode_btn.value()
        if mode_val == 0 and prev_mode_val == 1 \
                and time.ticks_diff(now_ms, last_mode_btn) > DEBOUNCE_MS:
            last_mode_btn = now_ms
            manual        = not manual
            shared["manual"][0] = manual
            if manual:
                locked  = False
                at_home = False
                shared["locked"][0] = False
                shared["fresh"][0]  = False
                stable_frames       = 0
                track_laser.off()
            else:
                last_ts = time.time()
                shared["yaw"][0]   = cur_yaw
                shared["pitch"][0] = cur_pitch
                shared["fresh"][0] = False
                stable_frames      = 0
        prev_mode_val = mode_val

        if fire_btn.value() == 0 and time.ticks_diff(now_ms, last_fire_btn) > DEBOUNCE_MS:
            last_fire_btn = now_ms
            if (not manual and locked) or manual:
                do_burst_fire(
                    track_laser, bullet_laser, buzzer,
                    servo_yaw, servo_pitch,
                    cur_yaw_ref, cur_pitch_ref,
                )
                cur_yaw               = cur_yaw_ref[0]
                cur_pitch             = cur_pitch_ref[0]
                stable_frames         = 0
                last_auto_fire_time   = time.time()
                if lcd:
                    try:
                        lcd.print_line("Target FIRED   ", 0)
                        lcd.print_line("  NEUTRALIZED  ", 1)
                    except: pass

        if manual:
            shared["fresh"][0] = False
            raw_pan  = adc_pan.read()
            raw_tilt = adc_tilt.read()
            smooth_pan  = smooth_pan  + JOY_ADC_SMOOTH * (raw_pan  - smooth_pan)
            smooth_tilt = smooth_tilt + JOY_ADC_SMOOTH * (raw_tilt - smooth_tilt)
            delta_yaw   = joy_to_delta(int(smooth_pan))
            delta_pitch = joy_to_delta(int(smooth_tilt))
            if delta_yaw != 0.0 or delta_pitch != 0.0:
                cur_yaw   = max(-YAW_LIMIT_DEG,   min(YAW_LIMIT_DEG,   cur_yaw   + delta_yaw))
                cur_pitch = max(-PITCH_LIMIT_DEG, min(PITCH_LIMIT_DEG, cur_pitch + delta_pitch))
                cur_yaw_ref[0]   = cur_yaw
                cur_pitch_ref[0] = cur_pitch
                servo_yaw.write(cur_yaw)
                servo_pitch.write(cur_pitch)

        else:
            if shared["cease_fire"][0]:
                if locked:
                    servo_yaw.write(0.0)
                    servo_pitch.write(0.0)
                    cur_yaw_ref[0]   = 0.0
                    cur_pitch_ref[0] = 0.0
                    cur_yaw   = 0.0
                    cur_pitch = 0.0
                    track_laser.off()
                    locked        = False
                    at_home       = True
                    stable_frames = 0
                    shared["locked"][0] = False

            elif shared["fresh"][0]:
                shared["fresh"][0] = False

                ty   = max(-YAW_LIMIT_DEG,   min(YAW_LIMIT_DEG,   shared["yaw"][0]))
                tp   = max(-PITCH_LIMIT_DEG, min(PITCH_LIMIT_DEG, shared["pitch"][0]))
                dist = min(shared["distance"][0], 50.0)
                last_ts = time.time()

                if not locked:
                    locked  = True
                    at_home = False
                    track_laser.on()
                    shared["locked"][0] = True
                    stable_frames = 0

                ey = ty - cur_yaw
                ep = tp - cur_pitch
                ey = 0.0 if abs(ey) < DEADBAND_DEG else ey
                ep = 0.0 if abs(ep) < DEADBAND_DEG else ep

                if ey or ep:
                    cur_yaw   += SMOOTH * ey
                    cur_pitch += SMOOTH * ep
                    cur_yaw_ref[0]   = cur_yaw
                    cur_pitch_ref[0] = cur_pitch
                    servo_yaw.write(cur_yaw)
                    servo_pitch.write(cur_pitch)

                residual_yaw   = abs(ty - cur_yaw)
                residual_pitch = abs(tp - cur_pitch)
                if residual_yaw <= AUTO_STABLE_YAW_DEG and residual_pitch <= AUTO_STABLE_PITCH_DEG:
                    stable_frames += 1
                else:
                    stable_frames = 0

                if (AUTO_FIRE_ENABLED
                        and shared["auto_fire"][0]
                        and locked
                        and not shared["cease_fire"][0]
                        and stable_frames >= AUTO_FIRE_STABLE_FRAMES
                        and not shared["firing"][0]):

                    fire_dist   = shared["us_dist"][0] if shared["us_dist"][0] > 0.0 else dist
                    now_t       = time.time()
                    cooldown_ok = (now_t - last_auto_fire_time) >= AUTO_FIRE_COOLDOWN_S

                    if AUTO_FIRE_MIN_M <= fire_dist <= AUTO_FIRE_MAX_M and cooldown_ok:
                        last_auto_fire_time = now_t
                        stable_frames       = 0
                        do_burst_fire(
                            track_laser, bullet_laser, buzzer,
                            servo_yaw, servo_pitch,
                            cur_yaw_ref, cur_pitch_ref,
                            label="[AUTO]",
                        )
                        cur_yaw   = cur_yaw_ref[0]
                        cur_pitch = cur_pitch_ref[0]
                        if lcd:
                            try:
                                lcd.print_line("AUTO-FIRE!     ", 0)
                                lcd.print_line("  NEUTRALIZED  ", 1)
                            except: pass

            else:
                silence = time.time() - last_ts
                if silence > TIMEOUT_HOME and not at_home:
                    servo_yaw.write(0.0)
                    servo_pitch.write(0.0)
                    cur_yaw             = 0.0
                    cur_pitch           = 0.0
                    cur_yaw_ref[0]      = 0.0
                    cur_pitch_ref[0]    = 0.0
                    track_laser.off()
                    locked              = False
                    at_home             = True
                    stable_frames       = 0
                    shared["locked"][0] = False

        gap = SERVO_INTERVAL_MS - time.ticks_diff(time.ticks_ms(), t0)
        if gap > 0:
            time.sleep_ms(gap)


try:
    main()
except KeyboardInterrupt:
    Pin(TRACK_LASER_PIN,  Pin.OUT).off()
    Pin(BULLET_LASER_PIN, Pin.OUT).off()
