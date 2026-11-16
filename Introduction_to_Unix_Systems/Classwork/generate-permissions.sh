#!/usr/bin/env bash

if [[ $# -ne 1 || ! $1 =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
    printf 'Usage: bash %s UNIVERSITY_LOGIN\n' "${0##*/}" >&2
    exit 2
fi

login=$1
workspace=unix-check02-permissions
answer_file=$workspace/permissions-$login.txt

octal_modes=(6745 6650 5720 5641 3750 3741 6520 6564 5510 3463)
symbolic_modes=(rwsr-S--- rwSr-sr-- rwsr-xr-T rwS-----t rwxrws--T rw-r-S--t r-sr-Sr-- r-S-ws--- rws---r-T -w-rwS--t)
masks=(002 007 013 027 033 046 071 137 226 352)

octal=${octal_modes[RANDOM % ${#octal_modes[@]}]}
symbolic=${symbolic_modes[RANDOM % ${#symbolic_modes[@]}]}
mask=${masks[RANDOM % ${#masks[@]}]}

umask 077
if ! mkdir -m 700 -- "$workspace"; then
    if [[ -f $answer_file ]]; then
        printf 'Use the worksheet you already generated at %s.\n' "$answer_file" >&2
    fi
    exit 1
fi

if ! cat > "$answer_file" <<EOF
Introduction to Unix systems
Classwork 02 | Task 02 | Permissions (4 points)
University login: $login

A. Convert the octal mode $octal to nine permission characters.
   Omit the file-type character.
   Symbolic mode:

B. Convert the symbolic mode $symbolic to four octal digits.
   Include the digit for the special bits.
   Octal mode:

C. The umask is $mask. Give the resulting modes in octal.
   Assume creation modes of 0666 for a regular file and 0777 for a
   directory, with no default special bits.
   New regular file:
   New directory:
EOF
then
    rm -f -- "$answer_file"
    rmdir -- "$workspace"
    exit 1
fi

printf 'Worksheet created at %s\n' "$answer_file"
