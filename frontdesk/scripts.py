"""Approved scripts: the fixed lines Front Desk speaks, word for word.

These never go through the model. Safety, compliance and handoff lines are
approved by the clinic, so the app speaks them exactly as written here, even
when the model is unavailable. Everything else the agent says is phrased by
the model (see ``frontdesk.model``).
"""

from __future__ import annotations

# FERN-13: the recording disclosure opens every call.
DISCLOSURE = "This call may be recorded and transcribed to help us serve you."
GREETING = (
    "Thank you for calling Brookside Family Clinic. This is Front Desk, the clinic's "
    "automated assistant. How can I help you today?"
)

# FERN-4: emergencies and medical questions.
EMERGENCY = "If this is a medical emergency, please hang up and call 911 now."
NURSE_LINE_OFFER = (
    "I can't give medical advice, but I can send a message to our nurse line and a nurse "
    "will call you back. Would you like me to do that?"
)
NURSE_MESSAGE_SENT = "I've sent a message to the nurse line. A nurse will call you back."

# FERN-8: handoff to a person.
TRANSFER = "I'm transferring you to a member of our staff now. Please hold."
TRANSFER_OFFER = "Would you like me to transfer you to a member of our staff?"
CALLBACK_OFFER = (
    "Our staff are available Monday to Friday, 8 AM to 5 PM, so I can't transfer you "
    "right now. I can have someone call you back."
)
CALLBACK_ASK_NUMBER = "What number should they call?"
CALLBACK_NUMBER_AGAIN = "Sorry, I didn't get a full ten-digit number. What number should they call?"
# The read-back is "I have <number>. Is that right?", with the number in say-as telephone.
CALLBACK_CONFIRM_BEFORE = "I have"
CALLBACK_CONFIRM_AFTER = "Is that right?"
CALLBACK_REQUESTED = (
    "Thank you. Someone from the clinic will call you back on the next business day."
)

# FERN-2: verification lockout.
VERIFY_LOCKED = "I'm sorry, I wasn't able to verify your identity."

# FERN-8: speech recognition trouble.
LOW_CONFIDENCE_REPROMPT = "Sorry, I didn't catch that. Could you say it again?"
LOW_CONFIDENCE_TROUBLE = "I'm having trouble hearing you."

# FERN-13: recording opt-out.
OPT_OUT_ACK = (
    "Understood. I've stopped recording, and I won't keep a transcript of the rest of this call."
)

# The model is down: approved fallback before a transfer offer.
MODEL_TROUBLE = "I'm sorry, I'm having trouble on my end right now."

# Keypad.
KEY_UNUSED = "Sorry, that key isn't used here. You can press 0 to reach our staff."

# Short lines that follow another reply.
OKAY = "Okay."
HOW_CAN_I_HELP = "How can I help?"
ANYTHING_ELSE = "Is there anything else I can help with?"
