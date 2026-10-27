"""Generate explicit UNIMPLEMENTED entry points for the prototype HAL subset."""

import pathlib
import re
import sys

sdk, implementation, output = map(pathlib.Path, sys.argv[1:])
implemented = set(
    re.findall(
        r"M_driver_status_t\s+(M_driver_\w+)\s*\(", implementation.read_text()
    )
)
text = [
    '// Generated from SDK declarations. Do not edit.\n#include "HAL/M_driver_api.h"\n'
]
seen = set()
for header in sorted((sdk / "HAL").glob("*.h")):
    for declaration in re.findall(
        r"M_DRIVER_API_EXPORT\s+(M_driver_status_t\s+M_driver_\w+\s*\(.*?\));",
        header.read_text(),
        re.DOTALL,
    ):
        name = re.search(r"(M_driver_\w+)\s*\(", declaration)[1]
        if name not in implemented and name not in seen:
            text.append(
                declaration + " { return M_DRIVER_STATUS_UNIMPLEMENTED; }\n"
            )
        seen.add(name)
output.write_text("\n".join(text))
