import {
  Children,
  createContext,
  forwardRef,
  isValidElement,
  type AnchorHTMLAttributes,
  type MouseEvent,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

export interface LocationState {
  pathname: string;
  search: string;
  hash: string;
  state: unknown;
}

interface NavigateOptions {
  replace?: boolean;
  state?: unknown;
}

type NavigateFunction = (to: string, options?: NavigateOptions) => void;

interface RouterValue {
  location: LocationState;
  navigate: NavigateFunction;
}

const RouterContext = createContext<RouterValue | null>(null);
const OutletContext = createContext<ReactNode>(null);
const ParamsContext = createContext<Record<string, string>>({});

function browserLocation(state: unknown = window.history.state): LocationState {
  return {
    pathname: window.location.pathname,
    search: window.location.search,
    hash: window.location.hash,
    state,
  };
}

function sameOriginTarget(to: string): string {
  const target = new URL(to, window.location.origin);
  if (target.origin !== window.location.origin) return "/";
  return `${target.pathname}${target.search}${target.hash}`;
}

export function BrowserRouter({ children }: { children: ReactNode }) {
  const [location, setLocation] = useState<LocationState>(() => browserLocation());
  const navigate = useCallback<NavigateFunction>((to, options = {}) => {
    const target = sameOriginTarget(to);
    const method = options.replace ? "replaceState" : "pushState";
    window.history[method](options.state ?? null, "", target);
    setLocation(browserLocation(options.state ?? null));
  }, []);

  useEffect(() => {
    const onPopState = (event: PopStateEvent) => setLocation(browserLocation(event.state));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const value = useMemo(() => ({ location, navigate }), [location, navigate]);
  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

function useRouter(): RouterValue {
  const value = useContext(RouterContext);
  if (!value) throw new Error("Router hooks must be used within BrowserRouter");
  return value;
}

export function useLocation(): LocationState {
  return useRouter().location;
}

export function useNavigate(): NavigateFunction {
  return useRouter().navigate;
}

export function useParams<T extends Record<string, string | undefined> = Record<string, string>>(): T {
  return useContext(ParamsContext) as T;
}

export interface RouteProps {
  path?: string;
  element?: ReactNode;
  children?: ReactNode;
}

export function Route(_props: RouteProps): null {
  return null;
}

interface RouteMatch {
  elements: ReactNode[];
  params: Record<string, string>;
}

function matchPattern(pattern: string, pathname: string): Record<string, string> | null {
  if (pattern === "*") return {};
  const names: string[] = [];
  const source = pattern
    .split("/")
    .map((segment) => {
      if (segment.startsWith(":")) {
        names.push(segment.slice(1));
        return "([^/]+)";
      }
      return segment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    })
    .join("/");
  const matched = new RegExp(`^${source}/?$`).exec(pathname);
  if (!matched) return null;
  return Object.fromEntries(names.map((name, index) => [name, decodeURIComponent(matched[index + 1])]));
}

function findRoute(children: ReactNode, pathname: string): RouteMatch | null {
  for (const child of Children.toArray(children)) {
    if (!isValidElement<RouteProps>(child) || child.type !== Route) continue;
    const { path, element, children: nestedChildren } = child.props;
    const ownParams = path ? matchPattern(path, pathname) : {};
    if (ownParams === null) continue;
    if (nestedChildren) {
      const nested = findRoute(nestedChildren, pathname);
      if (nested) {
        return {
          elements: element === undefined ? nested.elements : [element, ...nested.elements],
          params: { ...ownParams, ...nested.params },
        };
      }
    }
    if (path) return { elements: element === undefined ? [] : [element], params: ownParams };
  }
  return null;
}

function composeOutlets(elements: ReactNode[]): ReactNode {
  return elements.reduceRight<ReactNode>(
    (outlet, element) => <OutletContext.Provider value={outlet}>{element}</OutletContext.Provider>,
    null,
  );
}

export function Routes({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const match = findRoute(children, pathname);
  if (!match) return null;
  return <ParamsContext.Provider value={match.params}>{composeOutlets(match.elements)}</ParamsContext.Provider>;
}

export function Outlet() {
  return useContext(OutletContext);
}

export function Navigate({ to, replace = false, state }: { to: string; replace?: boolean; state?: unknown }) {
  const navigate = useNavigate();
  useEffect(() => navigate(to, { replace, state }), [navigate, replace, state, to]);
  return null;
}

export interface LinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> {
  to: string;
  replace?: boolean;
  state?: unknown;
}

export const Link = forwardRef<HTMLAnchorElement, LinkProps>(function Link(
  { to, replace = false, state, onClick, target, ...props },
  ref,
) {
  const navigate = useNavigate();
  const href = sameOriginTarget(to);
  const handleClick = (event: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(event);
    if (
      event.defaultPrevented
      || event.button !== 0
      || event.metaKey
      || event.ctrlKey
      || event.shiftKey
      || event.altKey
      || (target && target !== "_self")
    ) return;
    event.preventDefault();
    navigate(href, { replace, state });
  };
  return <a {...props} href={href} target={target} onClick={handleClick} ref={ref} />;
});

export const NavLink = Link;
