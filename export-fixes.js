(function () {
    "use strict";

    const SCHEMA_VERSION = 6;
    const PROFILE_FIELDS = ["ts", "fl"];

    function downloadJSON(filename, data) {
        const blob = new Blob(
            [JSON.stringify(data, null, 2)],
            { type: "application/json;charset=utf-8" }
        );

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = filename;
        link.style.display = "none";
        link.setAttribute("aria-hidden", "true");

        document.body.appendChild(link);

        try {
            link.click();
        } finally {
            setTimeout(function () {
                link.remove();
                URL.revokeObjectURL(url);
            }, 1000);
        }
    }

    function isValidCount(value) {
        return Number.isInteger(value) && value >= 0;
    }

    function validateServeCounts(values, label) {
        if (!Array.isArray(values) || values.length !== 6) {
            throw new Error(`${label} must contain exactly 6 values.`);
        }

        const counts = values.map(Number);

        if (!counts.every(isValidCount)) {
            throw new Error(
                `${label} contains invalid values. All counts must be nonnegative whole numbers.`
            );
        }

        const total = counts[0];
        const categoryTotal = counts.slice(1).reduce(
            (sum, count) => sum + count,
            0
        );

        if (total <= 0) {
            throw new Error(`${label} total serves must be greater than zero.`);
        }

        if (categoryTotal !== total) {
            throw new Error(
                `${label} category counts (${categoryTotal}) must equal total serves (${total}).`
            );
        }

        return counts;
    }

    function validateProfile(name, profile) {
        if (!name || !String(name).trim()) {
            throw new Error("A player profile has an empty name.");
        }

        if (!profile || typeof profile !== "object") {
            throw new Error(`Profile "${name}" is not a valid object.`);
        }

        const ts = validateServeCounts(profile.ts, `${name} Topspin data`);
        const fl = validateServeCounts(profile.fl, `${name} Float data`);

        return {
            ts: ts,
            fl: fl
        };
    }

    window.exportProfilesJSON = function () {
        try {
            const roster = typeof window.getRoster === "function"
                ? window.getRoster()
                : {};

            const cleaned = {};

            Object.entries(roster).forEach(function ([name, profile]) {
                cleaned[String(name)] = validateProfile(name, profile);
            });

            downloadJSON("coaches_volleyball_profiles.json", {
                version: SCHEMA_VERSION,
                exportedAt: new Date().toISOString(),
                players: cleaned
            });
        } catch (error) {
            console.error("Export error:", error);
            alert("Export failed: " + error.message);
        }
    };

    window.importProfilesJSON = function (event) {
        const input = event && event.target;
        const file = input && input.files ? input.files[0] : null;

        if (!file) return;

        const reader = new FileReader();

        reader.onload = function () {
            try {
                const parsed = JSON.parse(reader.result);
                const players = parsed.players || parsed.roster;

                if (!players || typeof players !== "object" || Array.isArray(players)) {
                    throw new Error("No player profiles were found.");
                }

                const cleaned = {};
                const errors = [];

                Object.entries(players).forEach(function ([name, profile]) {
                    try {
                        cleaned[String(name)] = validateProfile(name, profile);
                    } catch (error) {
                        errors.push(error.message);
                    }
                });

                if (errors.length > 0) {
                    throw new Error(errors.slice(0, 3).join(" "));
                }

                if (Object.keys(cleaned).length === 0) {
                    throw new Error("No valid player profiles were found.");
                }

                if (typeof window.saveRoster !== "function") {
                    throw new Error("Roster storage is unavailable.");
                }

                window.saveRoster(cleaned);

                alert(
                    "Loaded " +
                    Object.keys(cleaned).length +
                    " valid player profile(s) successfully."
                );
            } catch (error) {
                console.error("Import error:", error);
                alert("Import failed: " + error.message);
            } finally {
                input.value = "";
            }
        };

        reader.onerror = function () {
            alert("The selected file could not be read.");
            input.value = "";
        };

        reader.readAsText(file);
    };
})();
