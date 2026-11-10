// Empty string = relative URL, resolves against whatever host served this page.
// Only needed as an absolute URL for `npm start` dev server (set REACT_APP_API_URL
// in web_search_client/.env.development.local) since that runs on a different port than
// the API. Don't use .env.local: `npm run build` reads it too and would bake the URL in.
export const API_BASE = process.env.REACT_APP_API_URL || '';
