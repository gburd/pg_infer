#!/usr/bin/env bash
# ci/announce.sh VERSION
#
# Draft a PostgreSQL-style release announcement for pg_infer and submit it to
# the postgresql.org news/announce system (a login-authorized web form;
# approved news feeds the pgsql-announce list). Builds the announcement text
# from CHANGELOG.md + a fixed footer, then posts it using PGORG_* secrets. If
# those aren't fully set it prints the drafted announcement and exits 0, so a
# release never fails just because announcing isn't configured.
#
# NOTE (honest): the postgresql.org news system can only be *submitted* to via
# an authenticated two-step form and lands in a two-moderator queue -- it is
# NEVER auto-published. There is no way to auto-post to pgsql-announce; that
# gate is postgresql.org's. Even fully configured, this queues an article for
# human moderation. In the common case (secrets unset) it is a maintainer
# draft to paste into https://www.postgresql.org/account/news/new/ .
#
# Only feature (X.Y.0) and major (X.0.0) releases are announced. A patch
# release (X.Y.Z, Z>0) is skipped unless ANNOUNCE=1 or its CHANGELOG section
# has a "### Security" heading.
set -euo pipefail

VERSION="${1:?usage: announce.sh VERSION}"
VERSION="${VERSION#v}"
REPO_WEB="https://codeberg.org/gregburd/pg_infer"

# --- The changes for VERSION -------------------------------------------------
# pg_infer has no CHANGELOG, so the changes are the commit subjects between the
# previous tag and vVERSION (needs the tags: check out with fetch-depth: 0).
changelog_section() {
  local tag="v${VERSION}" prev
  git rev-parse -q --verify "refs/tags/${tag}" >/dev/null || return 0
  prev="$(git describe --tags --abbrev=0 "${tag}^" 2>/dev/null || true)"
  git log --no-merges --format='- %s' "${prev:+${prev}..}${tag}"
}
CHANGES="$(changelog_section)"

# --- Gate: announce feature/major releases, not trivial patch bumps -----------
patch="$(printf '%s\n' "$VERSION" | sed -nE 's/^[0-9]+\.[0-9]+\.([0-9]+).*/\1/p')"
if [ -n "$patch" ] && [ "$patch" -gt 0 ] && [ "${ANNOUNCE:-}" != 1 ] \
   && ! printf '%s\n' "$CHANGES" | grep -q '^### Security'; then
  echo "pg_infer ${VERSION} is a patch release without a '### Security' section; not announcing (ANNOUNCE=1 to force)."
  exit 0
fi

[ -n "$CHANGES" ] || CHANGES="See ${REPO_WEB}/commits/branch/main"

TITLE="pg_infer ${VERSION} released"
BODY="$(cat <<EOF
pg_infer ${VERSION} is now available.

pg_infer is a PostgreSQL extension that exposes what a transformer language
model "knows" -- its feature activations, learned associations and semantic
relationships -- as SQL-queryable relations, without running inference. It is
built on pgrx 0.17 for PostgreSQL 18. It is EXPERIMENTAL: the SQL API and the
vindex format may change between releases.

Changes in ${VERSION}:

${CHANGES}

pg_infer is not on PGXN. Build it from source with cargo pgrx (pgrx 0.17;
needs OpenSSL and OpenBLAS):
  git clone ${REPO_WEB}.git
  cd pg_infer && cargo pgrx install --pg-config /path/to/pg_config

Documentation is in the repository.
EOF
)"

# --- Submit (or dry-run) -----------------------------------------------------
# The postgresql.org news system (pgweb) requires: a password-login account
# (no OAuth/2FA), an APPROVED Organisation you manage, a confirmed org email,
# and one or more NewsTag ids. Two-step form: create the article
# (POST /account/news/new/) then confirm-submit (POST /account/news/<id>/submit/),
# after which two moderators must approve. CI cannot bootstrap the org/email/tags,
# so unless ALL of these are provided we print the drafted announcement and
# exit 0.
#   PGORG_USER, PGORG_PASSWORD  - a password-login postgresql.org account
#   PGORG_ORG_ID                - approved Organisation id (managed by that account)
#   PGORG_EMAIL_ID              - confirmed OrganisationEmail id (same org)
#   PGORG_TAGS                  - space-separated NewsTag ids (see
#                                 https://www.postgresql.org/about/news/taglist.json/)
BASE="https://www.postgresql.org"
if [ -z "${PGORG_USER:-}" ] || [ -z "${PGORG_PASSWORD:-}" ] \
   || [ -z "${PGORG_ORG_ID:-}" ] || [ -z "${PGORG_EMAIL_ID:-}" ] || [ -z "${PGORG_TAGS:-}" ]; then
  echo "PGORG_* not fully set (need USER, PASSWORD, ORG_ID, EMAIL_ID, TAGS) --"
  echo "printing the drafted announcement; submit it by hand at ${BASE}/account/news/new/ :"
  echo "----------------------------------------------------------------------"
  echo "Title: ${TITLE}"
  echo
  echo "${BODY}"
  echo "----------------------------------------------------------------------"
  exit 0
fi

CJ="$(mktemp)"; trap 'rm -f "$CJ"' EXIT
csrf() { grep -oE 'name="csrfmiddlewaretoken" value="[^"]+"' | head -1 | sed 's/.*value="//;s/"//'; }

# 1. Log in (Django LoginView).
LOGIN_CSRF="$(curl -sS -c "$CJ" "$BASE/account/login/" | csrf)"
curl -sS -c "$CJ" -b "$CJ" -e "$BASE/account/login/" -H "Origin: $BASE" \
  --data-urlencode "csrfmiddlewaretoken=$LOGIN_CSRF" \
  --data-urlencode "username=$PGORG_USER" \
  --data-urlencode "password=$PGORG_PASSWORD" \
  --data-urlencode "next=/account/news/new/" \
  "$BASE/account/login/" >/dev/null

# 2. Create the article (state CREATED). org/email/tags are required.
NEW_CSRF="$(curl -sS -c "$CJ" -b "$CJ" "$BASE/account/news/new/" | csrf)"
tagargs=(); for t in $PGORG_TAGS; do tagargs+=(--data-urlencode "tags=$t"); done
curl -sS -c "$CJ" -b "$CJ" -e "$BASE/account/news/new/" -H "Origin: $BASE" \
  --data-urlencode "csrfmiddlewaretoken=$NEW_CSRF" \
  --data-urlencode "org=$PGORG_ORG_ID" \
  --data-urlencode "email=$PGORG_EMAIL_ID" \
  --data-urlencode "title=$TITLE" \
  --data-urlencode "content=$BODY" \
  --data-urlencode "date=$(date +%F)" \
  "${tagargs[@]}" \
  "$BASE/account/news/new/" >/dev/null

# 3. Find the new article id, then confirm-submit it into the moderation queue.
ID="$(curl -sS -c "$CJ" -b "$CJ" "$BASE/account/edit/news/" \
      | grep -oE '/account/news/[0-9]+/' | grep -oE '[0-9]+' | sort -rn | head -1)"
if [ -z "$ID" ]; then
  echo "::warning::could not find the new article id; finish manually at $BASE/account/edit/news/"
  exit 0
fi
SUB_CSRF="$(curl -sS -c "$CJ" -b "$CJ" "$BASE/account/news/$ID/submit/" | csrf)"
code="$(curl -sS -o /tmp/pgorg_resp.txt -w '%{http_code}' -c "$CJ" -b "$CJ" \
  -e "$BASE/account/news/$ID/submit/" -H "Origin: $BASE" \
  --data-urlencode "csrfmiddlewaretoken=$SUB_CSRF" \
  --data-urlencode "confirm=on" \
  "$BASE/account/news/$ID/submit/")"
echo "news submit HTTP ${code} (article ${ID}) -- now pending moderation (two moderators approve)."
case "$code" in
  2*|3*) echo "Announcement queued." ;;
  *) echo "::warning::submit returned HTTP ${code}; finish manually at $BASE/account/edit/news/" ;;
esac
