import re
from string import punctuation
from typing import Optional


def remove_non_alphanum(s: str) -> str:
	return s.translate(str.maketrans("", "", punctuation)).replace(" ", "")


def validate_license_number(candidate: str) -> Optional[str]:
	if (
		not re.match(r"^[A-Z\d]{2}-[A-Z\d]{4}$", candidate)
		and not re.match(r"^[A-Z\d]{4}-[A-Z\d]{2}$", candidate)
		and not re.match(r"^[A-Z\d]{3}-[A-Z\d]{3}$", candidate)
		and not re.match(r"^[A-Z\d]{3}-[A-Z\d]{4}$", candidate)
	):
		return None

	return candidate
